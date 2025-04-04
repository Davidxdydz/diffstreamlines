from typing import Tuple
import torch
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
import diffstreamlines._C as _C


class Streamlines(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        velocities: torch.Tensor,
        start_positions: torch.Tensor,
        dt: float,
        steps: int,
    ) -> torch.Tensor:
        ctx.dt = dt
        ctx.width = velocities.shape[1]
        ctx.height = velocities.shape[0]
        paths, path_lengths = _C.streamlines_forward(
            velocities,
            start_positions,
            dt,
            steps,
        )
        ctx.save_for_backward(paths, path_lengths)
        return paths, path_lengths

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor, *args):
        # TODO why tf do i get args here
        paths, path_lengths = ctx.saved_tensors
        velocities_grad = _C.streamlines_backward(
            grad_output,
            paths,
            path_lengths,
            ctx.width,
            ctx.height,
            ctx.dt,
        )
        return velocities_grad, None, None, None


def streamlines(
    velocities: torch.Tensor,
    start_positions: torch.Tensor,
    dt: float,
    steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    H, W, D = velocities.shape
    assert D == 2, f"velocities must have shape (H,W,2), but got {velocities.shape}"
    N, D = start_positions.shape
    assert (
        D == 2
    ), f"start_positions must have shape (N,2), but got {start_positions.shape}"
    assert velocities.is_cuda, "velocities must be on GPU"
    assert start_positions.is_cuda, "start_positions must be on GPU"
    assert velocities.dtype == torch.float32, "velocities must be float32"
    assert start_positions.dtype == torch.float32, "start_positions must be float32"
    return Streamlines.apply(
        velocities,
        start_positions,
        dt,
        steps,
    )
