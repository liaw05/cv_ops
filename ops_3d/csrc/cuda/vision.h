// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include <torch/extension.h>


at::Tensor ROIAlign_forward_cuda(const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int pooled_depth,
                                 const int sampling_ratio);
// input: [n, c, h, w, d]
// rois columns: y1, x1, y2, x2, z1, z2, s

at::Tensor ROIAlign_backward_cuda(const at::Tensor& grad,
                                  const at::Tensor& rois,
                                  const int pooled_height,
                                  const int pooled_width,
                                  const int pooled_depth,
                                  const int batch_size,
                                  const int channels,
                                  const int height,
                                  const int width,
                                  const int depth,
                                  const int sampling_ratio);


at::Tensor nms_cuda(const at::Tensor boxes, float nms_overlap_thresh);
// boxes columns: y1, x1, y2, x2, z1, z2, s

