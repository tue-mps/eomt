import math
from math import prod

def _add_handler(inputs, outputs):
    """Number of FLOPs for an addition operation.

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shape = _get_cval_shape(outputs[0])
    # print(in_shapes, out_shape)
    # assert in_shapes[0][1:] == in_shapes[1][1:] and (in_shapes[0][0] == in_shapes[1][0] or in_shapes[1][0] == 1), \
    #     f"Got incompatible shapes for adding: {in_shapes}"
    return prod(out_shape)


def _mul_handler(inputs, outputs):
    """Number of FLOPs for a multiplication operation.

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shapes = _get_cval_shape(outputs)
    # assert len(in_shapes[1]) <= 1 or len(in_shapes[0]) <= 1 or in_shapes[1][1:] == [1, 1] or (len(in_shapes[0]) == len(in_shapes[1]) and all(x == y == out or (x == 1 and y == out) or (y == 1 and x == out) for x, y, out in zip(in_shapes[0], in_shapes[1], out_shapes[0]))), \
    #     f"mul_handler found in_shapes: {in_shapes} -> {out_shapes[0]}"
    # print(f"in: {in_shapes}\t->\tout: {out_shapes}")
    return prod(out_shapes[0])


def _softmax_handler(inputs, outputs):
    """Number of FLOPs for a softmax operation.

    approximate times 5 for flops from exp, sum, and mult (taken from https://github.com/google-research/electra/blob/master/flops_computation.py)

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shapes = _get_cval_shape(outputs)
    # print(f"in: {in_shapes}\t->\tout: {out_shapes}")

    # approximate times 5 for flops from exp, sum, and mult (taken from https://github.com/google-research/electra/blob/master/flops_computation.py)
    return prod(out_shapes[0]) * 5


def _gelu_handler(inputs, outputs):
    """Number of FLOPs for a gelu operation.

    approximate times * 8 for mult, add, tanh, and pow (taken from https://github.com/google-research/electra/blob/master/flops_computation.py)

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shape = _get_cval_shape(outputs[0])

    # approximate times * 8 for mult, add, tanh, and pow (taken from https://github.com/google-research/electra/blob/master/flops_computation.py)
    return prod(out_shape) * 8


def _div_handler(inputs, outputs):
    """Number of FLOPs for a division operation.

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shapes = _get_cval_shape(outputs)

    return prod(out_shapes[0])


def _norm_handler(inputs, outputs):
    """Number of FLOPs for a normalization operation.

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shapes = _get_cval_shape(outputs)[0]
    in_shapes = _get_cval_shape(inputs)[0]

    # flops come from squaring each input (M*N) and adding all of them up (M*N - 1)
    norm_dims = [1]
    batch_dims = [1]
    for dim in set(in_shapes):
        if dim == 1:
            continue
        in_cnt = in_shapes.count(dim)
        out_cnt = out_shapes.count(dim)
        assert in_cnt >= out_cnt, f"Found {dim} more in out shape ({out_shapes}) then in shape ({in_shapes})"
        batch_dims += [dim for _ in range(out_cnt)]
        norm_dims += [dim for _ in range(in_cnt - out_cnt)]

    return prod(batch_dims) * (2 * prod(norm_dims) - 1)


def _cumsum_handler(inputs, outputs):
    """Number of FLOPs for a cumsum operation.

    in cumsum_dim: 0 + 1 + ... + n-1 = n(n-1)/2
    for each of the batch dims (entries prod(all_dims) / cumsum_dim)

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shapes = _get_cval_shape(outputs)[0]
    in_shapes = _get_cval_shape(inputs)[0]
    assert out_shapes == in_shapes, f"cumsum: {out_shapes} != {in_shapes}"

    # assume worst case
    cumsum_dim = max(in_shapes)
    # in cumsum_dim: 0 + 1 + ... + n-1 = n(n-1)/2
    # for each of the batch dims (entries prod(all_dims) / cumsum_dim
    return int(prod(in_shapes) * (cumsum_dim - 1) / 2)


def _pow_handler(inputs, outputs):
    """Number of FLOPs for a power operation.

    assume pow <= 4 -> ~ 3 mults

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shapes = _get_cval_shape(outputs)[0]

    # print(f"pow map: {in_shapes} -> {out_shapes}")

    # for now assume pow <= 4 -> ~ 3 mults
    return 3 * prod(out_shapes)


def _sin_cos_handler(inputs, outputs):
    """Number of FLOPs for a sin/cos operation.

    approximate each of these operations (on GPU) to be just 1 FLOP
    taken from https://foldingathome.org/support/faq/flops/

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shape = _get_cval_shape(outputs)[0]

    # approximate each of these operations (on GPU) to be just 1 FLOP
    # taken from https://foldingathome.org/support/faq/flops/
    return prod(out_shape)


def _log_handler(inputs, outputs):
    """Number of FLOPs for a log operation.

    approximation operation costing 20 FLOPS
    taken from https://foldingathome.org/support/faq/flops/

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shape = _get_cval_shape(outputs)[0]

    # approximation operation costing 20 FLOPS
    # taken from https://foldingathome.org/support/faq/flops/
    return 20 * prod(out_shape)


def _exp_handler(inputs, outputs):
    """Number of FLOPs for an exp operation.

    approximation operation costing 20 FLOPS
    taken from https://foldingathome.org/support/faq/flops/

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shape = _get_cval_shape(outputs)[0]

    # approximation operation costing 20 FLOPS
    # taken from https://foldingathome.org/support/faq/flops/
    return 20 * prod(out_shape)


def _sigmoid_handler(inputs, outputs):
    """Number of FLOPs for a normalization operation.

    approximation: number of flops for exp + 2

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shape = _get_cval_shape(outputs)[0]

    # approximation: number of flops for exp + 2
    return 2 * prod(out_shape) + _exp_handler(inputs, outputs)


def _sum_handler(inputs, outputs):
    """Number of FLOPs for a summation operation.

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shape = _get_cval_shape(outputs)[0]
    in_shape = _get_cval_shape(inputs)[0]

    sum_dims = [1]
    batch_dims = [1]
    for dim in set(in_shape):
        if dim == 1:
            continue
        in_cnt = in_shape.count(dim)
        out_cnt = out_shape.count(dim)
        assert in_cnt >= out_cnt, f"Found {dim} more in out shape ({out_shape}) then in shape ({in_shape})"
        batch_dims += [dim for _ in range(out_cnt)]
        sum_dims += [dim for _ in range(in_cnt - out_cnt)]

    return prod(batch_dims) * (prod(sum_dims) - 1)


def _rfft2_handler(inputs, outputs):
    """Number of FLOPs for an FFT operation.

    FLOPS are approximate 2.5 * N * log_2(N) (taken from http://www.fftw.org/speed/method.html -> Cooley-Tukey algorithm)

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shape = _get_cval_shape(outputs)[0]
    in_shape = _get_cval_shape(inputs)[0]

    # assume w and h dims are next to each other
    # by default assume last two dimensions
    d_i_1, d_i_2 = -2, -1
    for i, (d_in, d_out) in enumerate(zip(in_shape, out_shape)):
        if d_in != d_out:
            d_i_1 = i
            d_i_2 = i - 1
            break

    # FLOPS are approximate 2.5 * N * log_2(N) (taken from http://www.fftw.org/speed/method.html -> Cooley-Tukey algorithm)
    N = in_shape[d_i_1] * in_shape[d_i_2]
    return int(prod(in_shape) * 2.5 * math.log2(N))


def _irfft2_handler(inputs, outputs):
    """Number of FLOPs for an inverse FFT operation.

    FLOPS are approximate 2.5 * N * log_2(N) (taken from http://www.fftw.org/speed/method.html -> Cooley-Tukey algorithm)

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shape = _get_cval_shape(outputs)[0]
    in_shape = _get_cval_shape(inputs)[0]

    # assume w and h dims are next to each other
    # by default assume last two dimensions
    d_i_1, d_i_2 = -2, -1
    for i, (d_in, d_out) in enumerate(zip(in_shape, out_shape)):
        if d_in != d_out:
            d_i_1 = i
            d_i_2 = i - 1
            break

    # FLOPS are approximate 2.5 * N * log_2(N) (taken from http://www.fftw.org/speed/method.html -> Cooley-Tukey algorithm)
    N = out_shape[d_i_1] * out_shape[d_i_2]
    return int(prod(out_shape) * 2.5 * math.log2(N))


def _fft2_handler(inputs, outputs):
    """Number of FLOPs for an FFT operation.

    FLOPS are approximate 5 * N * log_2(N) (taken from http://www.fftw.org/speed/method.html -> Cooley-Tukey algorithm)

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shape = _get_cval_shape(outputs)[0]
    in_shape = _get_cval_shape(inputs)[0]

    # assume w and h dims are next to each other
    # by default assume last two dimensions
    d_i_1, d_i_2 = -2, -1
    for i, (d_in, d_out) in enumerate(zip(in_shape, out_shape)):
        if d_in != d_out:
            d_i_1 = i
            d_i_2 = i - 1
            break

    # FLOPS are approximate 5 * N * log_2(N) (taken from http://www.fftw.org/speed/method.html -> Cooley-Tukey algorithm)
    N = out_shape[d_i_1] * out_shape[d_i_2]
    return int(prod(out_shape) * 5 * math.log2(N))


def _scaled_dot_product_attention_handler(inputs, outputs):
    qkv_shape = _get_cval_shape(inputs)[0]
    flops = prod(qkv_shape)  # scale q by sqrt d
    flops += prod(qkv_shape) * qkv_shape[-2] * 2  # Q x K^T matrix multiplication
    flops += prod(qkv_shape[:-1]) * qkv_shape[-2] * 5  # softmax calculation on QK^T (factor 5 from softmax operation)
    flops += prod(qkv_shape) * qkv_shape[-2] * 2  # A x V matrix multiplication
    return flops


def _mean_handler(inputs, outputs):
    """Number of FLOPs for a mean operation.

    mean of N elements takes N flops (N-1 for sum and 1 to divide by len)

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    in_shape = _get_cval_shape(inputs)[0]

    # mean of N elements takes N flops (N-1 for sum and 1 to divide by len)
    return prod(in_shape)


def _avg_pool2d_handler(inputs, outputs):
    """Number of FLOPs for an average pool operation.

    mean of N elements takes N flops (N-1 for sum and 1 to divide by len)

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    # take the mean; the same way as in mean_handler
    return _mean_handler(inputs, outputs)

def _get_cval_shape(val):
    """Get the shapes from a jit value object.

    Taken from https://github.com/facebookresearch/fvcore/blob/fd5043ff8b2e6790f5bd7c9632695c68986cc658/fvcore/nn/jit_handles.py#L23

    Args:
        val (torch._C.Value | list[torch._C.Value]): jit value object or list of those.

    Returns:
        list: shape

    """
    if isinstance(val, list):
        return [_get_cval_shape(x) for x in val]

    if val.isCompleteTensor():
        return val.type().sizes()
    return None

