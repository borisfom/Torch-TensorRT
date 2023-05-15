import operator
import warnings
from typing import Optional, cast
import math

import numpy as np

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
import torch
from torch.fx.node import Target

from torch_tensorrt.fx.types import TRTNetwork, TRTTensor
from torch_tensorrt.fx.converters.converter_utils import (
    SourceIR,
    set_layer_name,
    get_positive_dim,
    has_dynamic_shape,
)
from torch_tensorrt.fx.converters.impl.shape import (
    get_shape_with_dynamic_shape
)

def slice(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    start: int,
    stop: int,
    step: int,
    dim: int,
) -> TRTTensor:
    if not isinstance(input, TRTTensor):
        raise RuntimeError(
            f"slice_tensor received input {input} that is not part "
            "of the TensorRT region!"
        )

    ranks = len(input.shape) + (1 if network.has_implicit_batch_dimension else 0)
    dim = get_positive_dim(cast(int, dim), ranks)
    dynamic_shape = has_dynamic_shape(input.shape)
    if network.has_implicit_batch_dimension:
        if dim == 0:
            raise RuntimeError(
                f"We do not support slice_tensor at batch dim when it's implicit, got {dim}!"
            )
        dim = dim - 1
    else:
        if dynamic_shape:
            # Check whether slice target dim is dynamic shape dim
            assert input.shape[dim] != -1, "Can't chunk on dynamic shape dimension!"

    start_int = cast(int, start)
    stop_int = cast(int, stop)
    step_int = cast(int, step)
    start = [0] * len(input.shape)
    start[dim] = start_int
    stride = [1] * len(start)
    stride[dim] = step_int
    output_shape = list(input.shape)
    output_shape[dim] = math.ceil((stop_int - start_int) / step_int)

    if dynamic_shape > 0:
        output_shape = get_shape_with_dynamic_shape(
            network, output_shape, input_val, target, name
        )
    layer = network.add_slice(
        input,
        start=start,
        shape=[] if dynamic_shape else output_shape,
        stride=stride,
    )
    if dynamic_shape:
        layer.set_input(2, output_shape)
    set_layer_name(layer, target, name)
    return layer.get_output(0)

