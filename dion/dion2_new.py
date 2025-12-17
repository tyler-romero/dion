
import math
import torch
import torch.distributed as dist
from itertools import chain
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Callable, Generator, List, Optional, Tuple, Union

from .newton_schulz_triton import newton_schulz_triton, zeropower_via_newtonschulz5
from .opt_utils import (
    AsyncRuntime,
    AsyncTask,
    create_param_batches,
    pad_batch,
    to_local,
)
from .scalar_opts import adamw_update_foreach_async, lion_update_foreach_async

# Reuse Muon's helper functions
from .muon import (
    muon_update_newton_schulz,
    adjust_lr_spectral_norm,
    adjust_lr_rms_norm,
)

class Dion2(Optimizer):
    """
    Distributed Dion2 optimizer for PyTorch FSDP2. Also compatible with DDP.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
            Use DeviceMesh for FSDP2 and ProcessGroup for DistributedDataParallel.
        lr: Base learning rate. For Muon, this will be scaled based on the matrix dimensions.
            For element-wise update rules, this is the actual learning rate and no additional scaling is done.
        fraction: Fraction of submatrix to orthogonalize per update (0 < fraction <= 1).
        ef_decay: Error-feedback decay factor applied to selected submatrix.
        betas: Tuple of (beta1, beta2) for AdamW and Lion algorithms.
        weight_decay: Weight decay factor. 
        epsilon: Small value to avoid division by zero. 
        adjust_lr: How to adjust the learning rate for Muon updates ("spectral_norm" or "rms_norm" or None).
            "spectral_norm": Adjust based on spectral norm, for learning rate transfer across model scale.
            "rms_norm": Adjust based on RMS norm, for learning rate compatibility with Adam/AdamW.
            None: Do not adjust the learning rate.
        flatten: Whether to flatten 3D+ tensors to 2D for Muon updates.
            True: Tensors with 3+ dimensions are flattened to 2D. Use this for convolutional layers.
            False: Tensors are not flattened. 3D+ tensors are treated as batches of 2D matrices.
        use_triton: Whether to use Triton kernel for Newton-Schulz. Ignored if custom function is provided.
        newton_schulz_func: Use a custom Newton-Schulz function for orthogonalization.
            Signature is `func(input: Tensor, epsilon: float) -> Tensor`.
        verbose: Whether to print debug information during updates. This prints whether rows or columns are selected for submatrix selection process.

    Dion2 optimizer by Ahn et al.: TBD
    """ 

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        fraction: float = 0.25,
        ef_decay: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        adjust_lr: Optional[str] = "spectral_norm",
        flatten: bool = False,
        use_triton: bool = False,
        newton_schulz_func: Optional[Callable] = None,
        verbose: bool = False,
    ):
        # Validate hyperparameters
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if ef_decay < 0.0:
            raise ValueError(f"Invalid ef_decay: {ef_decay}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if adjust_lr not in ("spectral_norm", "rms_norm", None):
            raise ValueError(f"Invalid adjust_lr: {adjust_lr}")

        defaults = dict(
            lr=lr,
            ef_decay=ef_decay,
            fraction=fraction,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            epsilon=epsilon,
            flatten=flatten,
            adjust_lr=adjust_lr,
            algorithm="dion2",
            step=0,
        )
        super().__init__(params, defaults)

        # Distributed configuration
        if isinstance(distributed_mesh, DeviceMesh):
            if distributed_mesh.ndim != 1:
                raise ValueError(
                    f"Only 1D DeviceMesh supported, got {distributed_mesh.ndim}D."
                )
            self._device_rank = distributed_mesh.get_local_rank()
            self._world_size = distributed_mesh.size()
            self._process_group = distributed_mesh.get_group()
        elif isinstance(distributed_mesh, ProcessGroup):
            self._device_rank = dist.get_rank(distributed_mesh)
            self._world_size = dist.get_world_size(distributed_mesh)
            self._process_group = distributed_mesh
        elif distributed_mesh is None:
            self._device_rank = 0
            self._world_size = 1
            self._process_group = None
        else:
            raise TypeError(f"Invalid distributed_mesh type: {type(distributed_mesh)}")
        self._distributed_mesh = distributed_mesh

        # Newton-Schulz configuration
        if newton_schulz_func is not None:
            if not callable(newton_schulz_func):
                raise TypeError(f"newton_schulz_func must be callable")
            self._newton_schulz_func = newton_schulz_func
        elif use_triton:
            self._newton_schulz_func = newton_schulz_triton
        else:
            self._newton_schulz_func = zeropower_via_newtonschulz5
        self.verbose = verbose
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        dion2_groups = []
        lion_groups = []
        adamw_groups = []

        for group in self.param_groups:
            group["step"] += 1
            algo = group["algorithm"]
            if algo == "dion2":
                dion2_groups.append(group)
            elif algo == "lion":
                lion_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        dion2_tasks = self._create_dion2_tasks(dion2_groups, verbose=self.verbose)
        lion_tasks = self._create_lion_tasks(lion_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = chain(dion2_tasks, lion_tasks, adamw_tasks)
        runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=3)
        runtime.run()

        return loss

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        """Initialize optimizer state (identical to Muon)."""
        state = self.state[param]
        if not state:
            state["momentum"] = torch.zeros_like(param)
            if algo == "adamw":
                state["variance"] = torch.zeros_like(param)
        return state

    def _create_dion2_tasks(
        self, param_groups: List[dict], verbose: bool = False,
    ) -> Generator["AsyncTask", None, None]:
        """Create batched Dion2 update tasks."""
        for group in param_groups:
            assert group["algorithm"] == "dion2"
            assert all(p.ndim >= 2 for p in group["params"]), \
                "Dion2 only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            # Hyperparameters as tensors for torch.compile
            dion2_args = dict(
                lr=torch.tensor(group["lr"]),
                ef_decay=torch.tensor(group["ef_decay"]),
                fraction=group["fraction"],
                weight_decay=torch.tensor(group["weight_decay"]),
                epsilon=torch.tensor(group["epsilon"]),
                flatten=group["flatten"],
                adjust_lr=group["adjust_lr"],
                device_rank=self._device_rank,
                world_size=self._world_size,
                process_group=self._process_group,
                newton_schulz_func=self._newton_schulz_func,
            )

            # Batch parameters by world_size (same as Muon)
            for params in create_param_batches(group_params, batch_size=self._world_size):
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, "dion2") for p in params]
                momentums = [s["momentum"] for s in states]

                # Determine sharding configuration
                shard_dim = None
                is_batch_sharded = False

                if isinstance(params[0], DTensor):
                    if not isinstance(self._distributed_mesh, DeviceMesh):
                        raise RuntimeError(
                            "Must use DeviceMesh for DTensor parameters."
                        )

                    # Find sharded placements (skip size-1 mesh dims)
                    shard_placements = [
                        (i, p)
                        for i, p in enumerate(params[0].placements)
                        if p.is_shard() and params[0].device_mesh.size(i) > 1
                    ]

                    # Check for batch vs matrix dimension sharding
                    if not group["flatten"]:
                        matrix_dims = {params[0].ndim - 1, params[0].ndim - 2}
                        is_batch_sharded = any(
                            p.dim not in matrix_dims for _, p in shard_placements
                        )
                        shard_placements = [
                            (i, p) for i, p in shard_placements if p.dim in matrix_dims
                        ]

                    if len(shard_placements) == 1:
                        shard_dim = shard_placements[0][1].dim
                    elif len(shard_placements) > 1:
                        raise NotImplementedError(
                            "Multiple sharded dimensions not supported."
                        )

                    # Verify mesh alignment
                    if shard_placements:
                        mesh_dim = shard_placements[0][0]
                        if params[0].device_mesh.get_group(mesh_dim) != self._process_group:
                            raise RuntimeError("DTensor mesh doesn't match optimizer mesh.")

                # Handle batch-sharded 3D tensors (each device has different matrices)
                if is_batch_sharded:
                    for x, g, m in zip(params, gradients, momentums):
                        yield AsyncTask(
                            dion2_update_batch_async(
                                X=[x],
                                G=[g],
                                M=[m],
                                shard_dim=None,
                                **dion2_args,
                                verbose=verbose,
                            )
                        )
                else:
                    yield AsyncTask(
                        dion2_update_batch_async(
                            X=pad_batch(params, self._world_size),
                            G=pad_batch(gradients, self._world_size),
                            M=pad_batch(momentums, self._world_size),
                            shard_dim=shard_dim,
                            **dion2_args,
                            verbose=verbose,
                        )
                    )

    def _create_lion_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        """Create Lion update tasks."""
        for group in param_groups:
            assert group["algorithm"] == "lion"

            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue

            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, "lion") for p in params]
            momentums = [s["momentum"] for s in states]

            yield AsyncTask(
                lion_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    lr=torch.tensor(group["lr"]),
                    beta1=torch.tensor(group["beta1"]),
                    beta2=torch.tensor(group["beta2"]),
                    weight_decay=torch.tensor(group["weight_decay"]),
                )
            )

    def _create_adamw_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        """Create AdamW update tasks."""
        for group in param_groups:
            assert group["algorithm"] == "adamw"

            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue

            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, "adamw") for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]

            yield AsyncTask(
                adamw_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    V=to_local(variances),
                    lr=torch.tensor(group["lr"]),
                    beta1=torch.tensor(group["beta1"]),
                    beta2=torch.tensor(group["beta2"]),
                    weight_decay=torch.tensor(group["weight_decay"]),
                    step=torch.tensor(group["step"]),
                    epsilon=torch.tensor(group["epsilon"]),
                )
            )



# =============================================================================
# Core Dion2 Update Functions
# =============================================================================

def dion2_update_batch_async(
    X: List[Tensor],  # Parameters (DTensor or Tensor), padded to world_size
    G: List[Tensor],  # Gradients, padded to world_size
    M: List[Tensor],  # Momentum buffers (modified in place), padded to world_size
    lr: Tensor,
    ef_decay: Tensor,
    fraction: float,
    weight_decay: Tensor,
    epsilon: Tensor,
    flatten: bool,
    adjust_lr: Optional[str],
    device_rank: int,
    world_size: int,
    shard_dim: Optional[int] = None,
    process_group: Optional[ProcessGroup] = None,
    newton_schulz_func: Optional[Callable] = None,
    verbose: bool = False,
) -> Generator[None, None, None]:
    """
    Batched Dion2 update with fractional submatrix selection.
    
    Algorithm:
    1. Update momentum: M = M + G
    2. Select top-α fraction along select_dim by L2 norm, extract submatrix
    3. Apply ef_decay to selected slices in M
    4. Communicate and orthogonalize only the submatrix
    5. Apply weight update to corresponding slices
    
    Selection dimension (select_dim):
    - FSDP row-sharded (shard_dim=-2): select rows (select_dim=-2), row norms are local
    - FSDP col-sharded (shard_dim=-1): select cols (select_dim=-1), col norms are local  
    - DDP/Single GPU: select rows by default (select_dim=-2)
    
    Communication patterns:
    - FSDP (shard_dim is not None): 
        - All-to-all gathers selected slices to form full submatrix
        - Orthogonalize, then all-to-all scatter back
    - DDP (shard_dim is None, world_size > 1):
        - Each rank orthogonalizes one matrix from the batch
        - All-gather to distribute results
    - Single GPU: direct computation
    """
    assert len(X) == len(G) == len(M)

    # Determine selection dimension based on sharding
    # 
    # shard_dim from DTensor can be:
    #   - Absolute index (0, 1, 2, ...) 
    #   - Negative index (-2, -1)
    #   - None (not sharded)
    #
    # We need to map this to select_dim which is always -2 (rows) or -1 (cols)
    # relative to the last two dimensions (the matrix dimensions).
    #
    # For FSDP: select along the sharded dimension so norms are local
    # For DDP/Single-GPU: select along the SHORTER dimension to reduce Newton-Schulz compute
    
    ndim = X[0].ndim
    
    if shard_dim is not None:
        # Convert shard_dim to normalized form relative to tensor
        normalized_shard_dim = shard_dim if shard_dim < 0 else shard_dim - ndim
        
        # Check if shard_dim corresponds to a matrix dimension (last two dims)
        if normalized_shard_dim == -2:
            # Row-sharded: select rows, compute row norms (norm over cols)
            select_dim = -2
        elif normalized_shard_dim == -1:
            # Col-sharded: select cols, compute col norms (norm over rows)
            select_dim = -1
        else:
            # Batch dimension sharded: not a matrix dim, fall back to shorter dim
            num_rows, num_cols = X[0].shape[-2:]
            select_dim = -2 if num_rows <= num_cols else -1
    else:
        # DDP/Single-GPU: choose shorter dimension to reduce Newton-Schulz compute
        num_rows, num_cols = X[0].shape[-2:]
        select_dim = -2 if num_rows <= num_cols else -1

    # Debug: Print selection choice (only on first call per parameter shape)
    if verbose:
        _print_selection_choice(X[0].shape, shard_dim, select_dim, ndim)

    # Step 1: Update momentum and select top-α fraction along select_dim
    # All matrices in batch have identical shapes, enabling stacked operations
    U_selected, indices_list = dion2_pre_orthogonalize(
        G=to_local(G),
        M=to_local(M),
        fraction=fraction,
        ef_decay=ef_decay,
        select_dim=select_dim,
    )

    # Step 2: Communicate and orthogonalize selected submatrices
    # -------------------------------------------------------------------------
    # FSDP path: all-to-all
    # -------------------------------------------------------------------------
    if shard_dim is not None:
        assert len(X) == world_size
        assert process_group is not None
        assert isinstance(X[0], DTensor)
        
        recv_shards = [torch.empty_like(u) for u in U_selected]
        work = dist.all_to_all(recv_shards, U_selected, group=process_group, async_op=True)
        yield
        work.wait()

        # Concatenate along selection dimension to form full selected submatrix
        # select_dim matches shard_dim, so we concatenate along that dimension
        full_submatrix = torch.cat(recv_shards, dim=select_dim)

        # Orthogonalize the full selected submatrix
        full_submatrix = muon_update_newton_schulz(
            full_submatrix, newton_schulz_func, flatten=flatten, epsilon=epsilon
        )

        # Split back into shards along the same dimension
        send_shards = [
            t.contiguous()
            for t in torch.tensor_split(full_submatrix, world_size, dim=select_dim)
        ]

        # All-to-all: scatter orthogonalized shards back to original owners
        U_ortho = [torch.empty_like(u) for u in U_selected]
        work = dist.all_to_all(U_ortho, send_shards, group=process_group, async_op=True)
        yield
        work.wait()

    # -------------------------------------------------------------------------
    # DDP path: all-gather
    # -------------------------------------------------------------------------
    elif len(U_selected) > 1:
        assert len(U_selected) == world_size
        assert process_group is not None

        # This rank orthogonalizes the matrix at index device_rank
        my_submatrix = muon_update_newton_schulz(
            U_selected[device_rank], newton_schulz_func, flatten=flatten, epsilon=epsilon
        )

        # All-gather: collect orthogonalized submatrices from all ranks
        U_ortho = [torch.empty_like(u) for u in U_selected]
        work = dist.all_gather(
            U_ortho, my_submatrix.contiguous(), group=process_group, async_op=True
        )
        yield
        work.wait()

    # -------------------------------------------------------------------------
    # Single GPU path
    # -------------------------------------------------------------------------
    else:
        assert len(U_selected) == 1
        U_ortho = [
            muon_update_newton_schulz(
                U_selected[0], newton_schulz_func, flatten=flatten, epsilon=epsilon
            )
        ]

    # Step 3: Compute adjusted learning rate (based on full/global matrix shape)
    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape, flatten=flatten)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape, flatten=flatten)
    else:
        raise ValueError(f"Unknown adjust_lr: {adjust_lr}")

    # Step 4: Apply weight update to selected slices only
    dion2_post_orthogonalize(
        X=to_local(X),
        U_ortho=U_ortho,
        indices=indices_list,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
        select_dim=select_dim,
    )


# =============================================================================
# Optimized Pre-Orthogonalize Function (Stacked Operations)
# =============================================================================
#
# KEY INSIGHT: All matrices in a batch have identical shapes!
# This enables stacked/batched tensor operations instead of loops.
#
# SELECTION DIMENSION:
# - select_dim=-2 (rows): Compute row norms (norm over cols), select top-k rows
# - select_dim=-1 (cols): Compute col norms (norm over rows), select top-k cols
#
# For FSDP, select_dim matches shard_dim so norms are computed locally.
# For DDP/Single-GPU, select_dim is the shorter dimension to reduce compute.
#
# NORM CHOICE: L1 norm (sum of absolute values)
# - Cheaper than L2: no squaring or sqrt needed
# - Effective proxy for selecting high-magnitude rows/cols
#
# OPTIMIZATION 1: Stack into 3D tensor for batched ops
# ----------------------------------------------------
# Stack (N, rows, cols) enables:
#   - Single batched norm instead of N separate norms
#   - Single batched topk instead of N separate topk calls  
#   - Single batched gather instead of N separate index_selects
#
# OPTIMIZATION 2: foreach for gradient accumulation
# -------------------------------------------------
# Optimal for in-place batched additions.
# =============================================================================

@torch.compile(fullgraph=True)
def dion2_pre_orthogonalize(
    G: List[Tensor],
    M: List[Tensor],
    fraction: Tensor,
    ef_decay: Tensor,
    select_dim: int,
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Update momentum and select top-α fraction along select_dim.
    
    All matrices in the batch have identical shapes, enabling stacked operations.
    
    Args:
        G: List of gradients
        M: List of momentum buffers (modified in place)
        fraction: Fraction of rows/cols to select
        ef_decay: Decay factor for selected slices
        select_dim: Dimension to select along (-2 for rows, -1 for cols)
    
    For each matrix M (shape: rows x cols):
    1. M += G (accumulate gradient into momentum)
    2. Compute L1 norm along the OTHER dimension
       - select_dim=-2: norm over cols (dim=-1) → row norms
       - select_dim=-1: norm over rows (dim=-2) → col norms
    3. Select top-k indices where k = ceil(fraction * size_of_select_dim)
    4. Extract selected slices as submatrix
    5. Apply ef_decay to selected slices in M (in-place)
    
    Returns:
        U_selected: List of selected submatrices in bf16 for communication
        indices_list: List of selected indices for each matrix
    """
    dtype = M[0].dtype
    
    # Determine sizes and norm dimension
    # norm_dim is the dimension we compute norm OVER (the other dimension)
    # select_dim is the dimension we SELECT from
    num_select = M[0].size(select_dim)
    norm_dim = -1 if select_dim == -2 else -2
    k = max(1, int(math.ceil(fraction * num_select)))
    
    # OPTIMIZATION 1: foreach for batched gradient accumulation
    G_casted = [g.to(dtype=dtype) for g in G]
    torch._foreach_add_(M, G_casted)
    
    # OPTIMIZATION 2: Stack for batched norm and topk
    # Shape: (batch_size, rows, cols)
    M_stacked = torch.stack(M, dim=0)
    
    # Compute L1 norm along norm_dim (sum of absolute values)
    # - If select_dim=-2 (rows): norm over dim=-1 → shape (batch, rows)
    # - If select_dim=-1 (cols): norm over dim=-2 → shape (batch, cols)
    slice_norms = M_stacked.norm(p=1, dim=norm_dim)
    
    # Batched topk: indices shape (batch_size, k)
    _, indices = torch.topk(slice_norms, k, dim=-1, sorted=False)
    
    # OPTIMIZATION 3: Batched gather for slice extraction
    if select_dim == -2:
        # Selecting rows: expand indices to (..., k, cols)
        num_cols = M[0].size(-1)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, num_cols)
        selected_stacked = torch.gather(M_stacked, dim=-2, index=indices_expanded)
    else:
        # Selecting cols: expand indices to (..., rows, k)
        num_rows = M[0].size(-2)
        indices_expanded = indices.unsqueeze(-2).expand(-1, num_rows, -1)
        selected_stacked = torch.gather(M_stacked, dim=-1, index=indices_expanded)
    
    # Apply ef_decay to selected slices in original M tensors
    # Must loop because M tensors are separate (stack created a copy)
    # Use index_copy_ with proper dimension handling for arbitrary batch dims
    indices_list = list(indices.unbind(dim=0))
    for m, idx in zip(M, indices_list):
        # Extract, scale, and copy back using the correct dimension
        selected_slice = m.index_select(dim=select_dim, index=idx)
        m.index_copy_(dim=select_dim, index=idx, source=selected_slice * ef_decay)
    
    # Convert to bf16 and unstack for communication
    U_selected = list(selected_stacked.to(dtype=torch.bfloat16).unbind(dim=0))
    
    return U_selected, indices_list


# =============================================================================
# Optimized Post-Orthogonalize Function
# =============================================================================
#
# OPTIMIZATION 1: foreach for weight decay
# ----------------------------------------
# Single fused kernel for all X *= (1 - lr * wd) operations.
#
# OPTIMIZATION 2: Batched dtype conversion
# ----------------------------------------
# Convert all U tensors upfront for better memory planning.
#
# OPTIMIZATION 3: Loop with index_add_ (torch.compile optimized)
# --------------------------------------------------------------
# torch.compile fuses operations within each iteration.
# =============================================================================

@torch.compile(fullgraph=True)
def dion2_post_orthogonalize(
    X: List[Tensor],
    U_ortho: List[Tensor],
    indices: List[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
    select_dim: int,
):
    """
    Apply weight decay (to all elements) and update selected slices only.
    
    Args:
        X: List of parameters to update
        U_ortho: List of orthogonalized update submatrices
        indices: List of selected indices for each matrix
        base_lr: Base learning rate (for weight decay)
        adjusted_lr: Adjusted learning rate (for updates)
        weight_decay: Weight decay factor
        select_dim: Dimension that was selected (-2 for rows, -1 for cols)
    
    Weight decay: X = X * (1 - base_lr * weight_decay)  [all elements]
    Update: X[selected_slices] -= adjusted_lr * U_ortho [selected slices only]
    """
    # OPTIMIZATION 1: foreach for batched weight decay
    torch._foreach_mul_(X, 1 - base_lr * weight_decay)
    
    # OPTIMIZATION 2: Batch dtype conversion upfront
    dtype = X[0].dtype
    U_converted = [u.to(dtype=dtype) for u in U_ortho]
    
    # OPTIMIZATION 3: Precompute scaled updates
    neg_lr = -adjusted_lr
    U_scaled = [neg_lr * u for u in U_converted]
    
    # Apply updates to selected slices
    # torch.compile optimizes this loop
    for x, u_scaled, idx in zip(X, U_scaled, indices):
        x.index_add_(dim=select_dim, index=idx, source=u_scaled)

 

 




# =============================================================================
# Debug Helper: Print Selection Choice (once per configuration)
# =============================================================================

_printed_configs: set = set()

def _print_selection_choice(
    shape: torch.Size, 
    shard_dim: Optional[int], 
    select_dim: int,
    ndim: int,
):
    """Print the selection dimension choice once per unique configuration."""
    config_key = (tuple(shape), shard_dim, select_dim)
    if config_key not in _printed_configs:
        _printed_configs.add(config_key)
        
        num_rows, num_cols = shape[-2:]
        select_info = "rows" if select_dim == -2 else "columns"
        norm_info = "row norms" if select_dim == -2 else "col norms"
        
        if shard_dim is None:
            mode = "DDP/Single-GPU"
            shorter = "rows" if num_rows <= num_cols else "cols"
            reason = f"shorter dim = {shorter} ({min(num_rows, num_cols)})"
        else:
            # Normalize shard_dim for display
            normalized = shard_dim if shard_dim < 0 else shard_dim - ndim
            if normalized == -2:
                mode = "FSDP"
                reason = f"row-sharded (shard_dim={shard_dim}→-2)"
            elif normalized == -1:
                mode = "FSDP"
                reason = f"col-sharded (shard_dim={shard_dim}→-1)"
            else:
                mode = "FSDP batch-sharded"
                shorter = "rows" if num_rows <= num_cols else "cols"
                reason = f"shard_dim={shard_dim} (batch), shorter = {shorter}"
        
        print(f"[Dion2] Shape {tuple(shape)}: {mode}, {reason} → "
              f"select top-α {select_info} by {norm_info}")
