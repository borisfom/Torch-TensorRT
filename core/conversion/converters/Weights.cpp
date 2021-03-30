#include "core/conversion/converters/Weights.h"
#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {

Weights::Weights() {
  this->num_input_maps = 0;
  this->num_output_maps = 0;
  this->data.type = nvinfer1::DataType::kFLOAT;
  this->data.values = nullptr;
  this->data.count = 0;
}

Weights::Weights(ConversionCtx* ctx, float val) {
  this->num_input_maps = 1;
  this->num_output_maps = 1;

  this->data.type = nvinfer1::DataType::kFLOAT;
  float* buf = reinterpret_cast<float*>(malloc(1 * sizeof(float)));
  buf[0] = val;
  this->data.values = buf;
  this->data.count = 1;
  ctx->builder_resources.push_back(buf);

  this->shape.nbDims = 0;
  this->kernel_shape.nbDims = 0;
}

Weights::Weights(ConversionCtx* ctx, int32_t val) {
  this->num_input_maps = 1;
  this->num_output_maps = 1;

  this->data.type = nvinfer1::DataType::kINT32;
  int32_t* buf = reinterpret_cast<int32_t*>(malloc(1 * sizeof(int32_t)));
  buf[0] = val;
  this->data.values = buf;
  this->data.count = 1;
  ctx->builder_resources.push_back(buf);

  this->shape.nbDims = 0;
  this->kernel_shape.nbDims = 0;
}

void
Weights::init(ConversionCtx* ctx,   nvinfer1::Dims dims,  nvinfer1::DataType dtype_optional, void* t_cpu, int numel) {
    int32_t ndims = dims.nbDims;
        if (ndims > nvinfer1::Dims::MAX_DIMS) {
            TRTORCH_THROW_ERROR(
                    "The tensor requested to be converted to nvinfer1::Weights exceeds the max number of dimensions for TensorRT");
        }
        this->shape = dims;
        if (ndims >= 2) {
            // Linear and Conv2/3D
            this->num_input_maps = dims.d[1];
            this->num_output_maps = dims.d[0];
        } else {
            // Bias
            this->num_input_maps = dims.d[0];
            this->num_output_maps = dims.d[0];
        }

        if (ndims > 2) {
            this->kernel_shape.nbDims = ndims - 2;

            for (size_t i = 2; i < ndims; i++) {
                this->kernel_shape.d[i - 2] = dims.d[i];
            }
        } else {
            this->kernel_shape.nbDims = 1;
            this->kernel_shape.d[0] = 1;
        }

        // Store the data in the conversion context so it remains until building is
        // complete

        void* buf = nullptr;
        int typesize = 0;
        switch (dtype_optional) {
            case nvinfer1::DataType::kFLOAT:
                typesize = sizeof(float);
                break;
            case nvinfer1::DataType::kINT32:
                typesize = sizeof(float);
                break;
            case nvinfer1::DataType::kHALF:
                typesize = 2;
                break;
            case nvinfer1::DataType::kINT8:
                typesize = 1;
                break;
            case nvinfer1::DataType::kBOOL:
                typesize = sizeof(bool);
                break;
            default:
            TRTORCH_THROW_ERROR("Found unsupported data type for tensor to weight conversion");
                break;
        }

        buf = malloc(numel * typesize);
        memcpy(buf, t_cpu, numel * typesize);
        ctx->builder_resources.push_back(buf);

        this->data.type = dtype_optional;
        this->data.count = numel;
        this->data.values = buf;

        LOG_DEBUG(*this);
    }
Weights::Weights(ConversionCtx* ctx, at::Tensor t) {

    auto dims = util::toDims(t.sizes());
    auto t_cpu = t.to(at::kCPU);
    t_cpu = t_cpu.contiguous();
    auto dtype_optional = util::toTRTDataType(t_cpu.dtype());
    if (!dtype_optional) {
        TRTORCH_THROW_ERROR("The tensor requested to be converted to nvinfer1::Weights is of an unsupported type");
    }
    init(ctx, dims, dtype_optional.value(), t_cpu.data_ptr(), t_cpu.numel());
}

Weights::Weights(ConversionCtx* ctx, nvinfer1::ITensor* t) {
    auto dims = t->getDimensions();
    nvinfer1::TensorLocation loc = t->getLocation();
    auto dtype_optional = t->getType();
    if (!dtype_optional) {
        TRTORCH_THROW_ERROR("The tensor requested to be converted to nvinfer1::Weights is of an unsupported type");
    }
    init(ctx, dims, dtype_optional.value(), t->data_ptr(), t_cpu.numel());
}
// clang-format off
std::ostream& operator<<(std::ostream& os, const Weights& w) {
  os << "Weights: " << w.shape
     << "\n    Number of input maps: " << w.num_input_maps
     << "\n    Number of output maps: " << w.num_output_maps
     << "\n    Element shape: [";
  for (int i = 0; i < w.kernel_shape.nbDims; i++) {
    os << w.kernel_shape.d[i];
    if (i + 1 < w.kernel_shape.nbDims) {
      os << ',';
    }
  }
  os << ']';
  return os;
}
// clang-format on
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
