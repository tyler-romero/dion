import torch


@torch.compile(fullgraph=True)
def polar_express5(G: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    # https://github.com/karpathy/nanochat/blob/16b8ac7da33010dc7efcd9292f895703f5cff33a/nanochat/optim.py#L110-L123
    polar_express_coeffs = [
        (8.156554524902461, -22.48329292557795, 15.878769915207462),
        (4.042929935166739, -2.808917465908714, 0.5000178451051316),
        (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
        (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
        (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
    ]

    assert G.ndim >= 2
    X = G.bfloat16()  # for speed
    if G.size(-2) > G.size(-1):
        X = X.mT  # this reduces FLOPs

    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + epsilon)

    for a, b, c in polar_express_coeffs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X  # X <- aX + bX^3 + cX^5

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X
