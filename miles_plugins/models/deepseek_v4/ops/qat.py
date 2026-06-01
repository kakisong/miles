import torch
from .kernel.act_quant import act_quant


def fp8_simulate(x: torch.Tensor, block_size: int):
    return act_quant(x.contiguous(), block_size, scale_fmt=None, scale_dtype=torch.float32, inplace=True)


class DeepSeekV4LinearQATFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, block_size=128):
        return fp8_simulate(kv, block_size)

    @staticmethod
    def backward(ctx, grad_kv):
        return grad_kv, None


fp8_simulate_qat = DeepSeekV4LinearQATFunc.apply
