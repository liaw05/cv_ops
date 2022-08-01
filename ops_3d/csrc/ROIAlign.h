#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

// Interface for Python
at::Tensor ROIAlign_forward(const at::Tensor& input,
                            const at::Tensor& rois,
                            const int pooled_height,
                            const int pooled_width,
                            const int pooled_depth,
                            const int sampling_ratio) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return ROIAlign_forward_cuda(input, rois, pooled_height, pooled_width, pooled_depth, sampling_ratio);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return ROIAlign_forward_cpu(input, rois, pooled_height, pooled_width, pooled_depth, sampling_ratio);
}

at::Tensor ROIAlign_backward(const at::Tensor& grad,
                             const at::Tensor& rois,
                             const int pooled_height,
                             const int pooled_width,
                             const int pooled_depth,
                             const int batch_size,
                             const int channels,
                             const int height,
                             const int width,
                             const int depth,
                             const int sampling_ratio) {
  if (grad.type().is_cuda()) {
#ifdef WITH_CUDA
    return ROIAlign_backward_cuda(grad, rois, pooled_height, pooled_width, pooled_depth, batch_size, channels, height, width, depth, sampling_ratio);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

