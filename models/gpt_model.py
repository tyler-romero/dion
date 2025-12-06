import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from typing import Optional


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 2
    n_head: int = 6
    n_embd: int = 768


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = None
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def init_inv_freq(self):
        # This needs to be a separate function so we can initialize model on torch.device("meta")
        self.inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )

    def forward(self, x):
        assert self.inv_freq is not None, "inv_freq not initialized"
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(
            k, (k.size(-1),)
        )  # QK norm suggested by @Grad62304977
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, -1)
        )  # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(
            x
        ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

        elif isinstance(module, Rotary):
            module.init_inv_freq()

    def forward(self, idx, targets=None, return_logits=False):
        x = self._forward_emb(idx)
        return self._forward(x, targets, return_logits)

    def _forward_emb(self, idx):
        # forward pass for just the embedding layer
        return self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

    def _forward(self, x, targets, return_logits):
        # forward pass for the rest of the model
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float()  # use tf32/fp32 for logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            logits = logits.float()  # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if return_logits:
            return logits
        else:
            return loss

    def compile(self):
        # Workaround for issue where torch.compile fails for embedding layer
        # Compile embedding separately from rest of the model
        # https://github.com/pytorch/torchtitan/issues/534
        self._forward = torch.compile(self._forward)
        self._forward_emb = torch.compile(self._forward_emb)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Repeatedly call self.forward(..., return_logits=True) to get next‑token logits,
        then sample and append.  Keeps only the last `sequence_len` tokens as context.
        """
        idx = input_ids
        for _ in range(max_new_tokens):
            # crop context if too long
            if idx.size(1) > self.config.sequence_len:
                idx = idx[:, -self.config.sequence_len :]

            # forward to get logits for every position
            logits = self(idx, targets=None, return_logits=True)
            # pick the logits for the very last position
            next_logits = logits[:, -1, :] / temperature

            # optionally top‑k filter
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # shape (B,1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


def parallelize_gpt_model(
    model: GPT,
    device_mesh: DeviceMesh,
    dp_name: Optional[str] = "dp",
    fs_name: Optional[str] = "fs",
    tp_name: Optional[str] = "tp",
    fsdp_reshard_after_forward: bool = True,
):
    """
    Parallelize GPT model using the given device mesh and sharding axis names.
    The model is modified in place.

    dp_name: Name of the mesh dimension to apply (replicated) data parallel
    fs_name: Name of the mesh dimension to apply fully sharded data parallel
    tp_name: Name of the mesh dimension to apply tensor parallel
    """
    # Get mesh dimensions to be used
    required_mesh_names = [x for x in (dp_name, fs_name, tp_name) if x]
    target_ndim = len(required_mesh_names)
    if target_ndim == 0:
        raise ValueError(
            "At least one of dp_name, fs_name, or tp_name must be provided"
        )

    # DP requires FS
    if dp_name and not fs_name:
        raise ValueError("Data parallelism with fully_shard() requires 2D FSDP mesh")

    # Check that mesh has correct number of dimensions
    if device_mesh.ndim < target_ndim:
        raise ValueError(
            f"Expected {target_ndim}-D device mesh {required_mesh_names}, but got mesh with {device_mesh.ndim} dimensions"
        )

    # Check that actual mesh names match the expected names
    actual_names = list(device_mesh.mesh_dim_names)
    if not all(name in actual_names for name in required_mesh_names):
        raise ValueError(
            f"Expected device mesh to have names {required_mesh_names}, but got {actual_names}"
        )

    # Apply TP
    # Keep track of whether TP is enabled so that we set shard placement differently for FSDP
    tp_enabled = False
    if tp_name:
        tp_mesh = device_mesh[tp_name]
        if tp_mesh.size() > 1:
            _apply_tp(model, tp_mesh)
            tp_enabled = True
    # Apply FSDP (while keeping track of whether TP is enabled)
    if fs_name:
        fsdp_mesh = (
            device_mesh[fs_name] if not dp_name else device_mesh[dp_name, fs_name]
        )
        _apply_fsdp(model, fsdp_mesh, fsdp_reshard_after_forward, tp_enabled=tp_enabled)


def _apply_tp(model: GPT, tp_mesh: DeviceMesh):
    # Apply TP to embedding and lm_head
    # Shard weights to save memory but replicate both inputs and outputs
    tp_plan = {
        # RowwiseParallel for nn.Embedding will do Shard(0)
        "transformer.wte": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Replicate(),
        ),
        # ColwiseParallel for nn.Linear will do Shard(0)
        "lm_head": ColwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Replicate(),
        ),
    }
    parallelize_module(
        model,
        tp_mesh,
        parallelize_plan=tp_plan,
    )

    # Apply TP to transformer layers
    for block in model.transformer.h:
        # Apply TP
        tp_plan = {
            "attn.c_q": ColwiseParallel(),
            "attn.c_k": ColwiseParallel(),
            "attn.c_v": ColwiseParallel(),
            "attn.c_proj": RowwiseParallel(),
            "mlp.c_fc": ColwiseParallel(),
            "mlp.c_proj": RowwiseParallel(),
        }
        parallelize_module(
            block,
            tp_mesh,
            parallelize_plan=tp_plan,
        )

        # Adjust number of attention heads for TP
        assert (
            block.attn.n_head % tp_mesh.size() == 0
        ), f"attn.n_head {block.attn.n_head} must be divisible by TP size {tp_mesh.size()}"
        block.attn.n_head = block.attn.n_head // tp_mesh.size()


def _apply_fsdp(
    model: GPT,
    fsdp_mesh: DeviceMesh,
    reshard_after_forward: bool = True,
    tp_enabled: bool = False,
):
    # FSDP mixed precision
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )

    # FSDP applied bottom-up starting with individual transformer blocks
    for block in model.transformer.h:
        # Apply DP and FS as fully_shard() hybrid sharding

        # Default shard placement when TP is disabled
        shard_placement_fn = None

        # Shard placement when TP is enabled
        if tp_enabled:
            # Shard the opposite dimension as TP
            # TP ColwiseParallel = Shard(0), so FSDP should use Shard(1)
            # TP RowwiseParallel = Shard(1), so FSDP should use Shard(0)
            shard_map = {
                block.attn.c_q.weight: Shard(1),
                block.attn.c_k.weight: Shard(1),
                block.attn.c_v.weight: Shard(1),
                block.attn.c_proj.weight: Shard(0),
                block.mlp.c_fc.weight: Shard(1),
                block.mlp.c_proj.weight: Shard(0),
            }
            shard_placement_fn = lambda param: shard_map.get(param)

        fully_shard(
            block,
            mesh=fsdp_mesh,
            shard_placement_fn=shard_placement_fn,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )

    # Apply DP and FS to embedding and lm_head
    # Don't reshard after forward since backward happens immediately afterwards

    # Default shard placement when TP is disabled
    shard_placement_fn = None

    # Shard placement when TP is enabled
    if tp_enabled:
        shard_map = {
            model.transformer.wte.weight: Shard(1),
            model.lm_head.weight: Shard(1),
        }
        shard_placement_fn = lambda param: shard_map.get(param)
    fully_shard(
        model,
        mesh=fsdp_mesh,
        mp_policy=mp_policy,
        shard_placement_fn=shard_placement_fn,
        reshard_after_forward=False,
    )
