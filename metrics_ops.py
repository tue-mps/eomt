from metrics_handlers import (
    _add_handler,       
    _mul_handler,
    _softmax_handler,
    _gelu_handler,
    _div_handler,
    _norm_handler,
    _cumsum_handler,
    _pow_handler,
    _sin_cos_handler,
    _log_handler,
    _exp_handler,
    _sigmoid_handler,
    _sum_handler,
    _rfft2_handler,
    _irfft2_handler,
    _fft2_handler,
    _mean_handler,
    _avg_pool2d_handler,
    _scaled_dot_product_attention_handler
)
import traceback

from contextlib import nullcontext

import torch.cuda
from fvcore.nn import FlopCountAnalysis
from loguru import logger
from torchprofile import profile_macs

def macs(args, model, input, n_ims=1):
    """Calculate the MACs (multiply-accumulate operations) of the model for a given input.

    Args:
        args: training arguments
        n_ims (int, optional): number of images to look at (Default value = 1)
        model (torch.nn.Module): the model
        input (torch.Tensor): the input tensor in batch format

    Returns:
        int: the number of MACs

    """
    if n_ims is not None:
        input = input[:n_ims]
    with torch.amp.autocast("cuda") if args.eval_amp and args.cuda else nullcontext():
        return profile_macs(model, input)


def flops(args, model, input, per_module=False, n_ims=1):
    """Return the number of floating point operations (FLOPs) needed for a given input.

    This function is broken, when working with timm models -> returns 0.
    The output should in theory be 2*MACs(), but it might report MACs straight up...
    Further investigation needed.

    Args:
        args: training arguments; in particular set args.eval_amp
        n_ims (int, optional): number of images to look at (Default value = 1)
        model (torch.nn.Module): the model to analyze
        input (torch.Tensor): the input to give to the model
        per_module (bool, optional): flag to return stats by submodule (Default value = False)

    Returns:
        int | dict: the number of FLOPs or a dictionary of FLOPs per submodule

    """
    if n_ims is not None:
        input = input[:n_ims]

    fca = FlopCountAnalysis(model, input)
    fca.set_op_handle("aten::add", _add_handler)
    fca.set_op_handle("aten::add_", _add_handler)
    fca.set_op_handle("aten::mul", _mul_handler)
    fca.set_op_handle("aten::mul_", _mul_handler)
    fca.set_op_handle("aten::softmax", _softmax_handler)
    fca.set_op_handle("aten::gelu", _gelu_handler)
    fca.set_op_handle("aten::bernoulli_", None)
    fca.set_op_handle("aten::div_", _div_handler)
    fca.set_op_handle("aten::div", _div_handler)
    fca.set_op_handle("aten::norm", _norm_handler)
    fca.set_op_handle("aten::cumsum", _cumsum_handler)
    fca.set_op_handle("aten::pow", _pow_handler)
    fca.set_op_handle("aten::sin", _sin_cos_handler)
    fca.set_op_handle("aten::cos", _sin_cos_handler)
    fca.set_op_handle("aten::sum", _sum_handler)
    fca.set_op_handle("aten::fft_rfft2", _rfft2_handler)
    fca.set_op_handle("aten::fft_irfft2", _irfft2_handler)
    fca.set_op_handle("aten::fft_fft2", _fft2_handler)
    fca.set_op_handle("aten::mean", _mean_handler)
    fca.set_op_handle("aten::sub", _add_handler)
    fca.set_op_handle("aten::rsub", _add_handler)
    fca.set_op_handle("aten::reciprocal", _div_handler)
    fca.set_op_handle("aten::avg_pool2d", _avg_pool2d_handler)
    fca.set_op_handle("aten::adaptive_avg_pool1d", _avg_pool2d_handler)
    fca.set_op_handle("aten::log", _log_handler)
    fca.set_op_handle("aten::exp", _exp_handler)
    fca.set_op_handle("aten::sigmoid", _sigmoid_handler)
    fca.set_op_handle("aten::scatter_add", _add_handler)
    fca.set_op_handle("aten::log_softmax", _softmax_handler)
    fca.set_op_handle("aten::square", _mean_handler)
    fca.set_op_handle("aten::scaled_dot_product_attention", _scaled_dot_product_attention_handler)

    # these operations are ignored, because 0 FLOPS
    fca.set_op_handle("aten::expand_as", None)
    fca.set_op_handle("aten::clamp_min", None)
    fca.set_op_handle("aten::view_as_complex", None)
    fca.set_op_handle("aten::real", None)
    fca.set_op_handle("aten::eye", None)
    fca.set_op_handle("aten::repeat_interleave", None)
    fca.set_op_handle("aten::scatter_reduce", None)
    fca.set_op_handle("aten::fill_", None)
    fca.set_op_handle("aten::ones_like", None)
    fca.set_op_handle("aten::topk", None)
    fca.set_op_handle("aten::expand", None)
    fca.set_op_handle("aten::reshape", None)
    fca.set_op_handle("aten::permute", None)
    fca.set_op_handle("aten::unbind", None)

    with torch.amp.autocast("cuda") if args.eval_amp and args.cuda else nullcontext():
        if per_module:
            return fca.by_module()
        try:
            return fca.total()
        except IndexError as e:
            logger.error(f"IndexError {e} when calculating flops. Might come from timm model.")
            traceback.print_exc()
            return -1
