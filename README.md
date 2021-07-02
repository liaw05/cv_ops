# CV Ops

This is a PyTorch implementation of 2d/3d nms, roi align, deform pool and deform conv for cpu and cuda.

### Building
For 2d Ops
```
cd ops_2d
sudo python3 build.py build_ext develop
```

For 3d Ops
```
cd ops_3d
sudo python3 build.py build_ext develop
```

### Usage
Must **import torch** before **import torch_op_2d or import torch_op_3d**.
```
import torch
import torch_op_2d
import torch_op_3d
```

Directly use 3d nms function
```
from layers import nms
nms.nms_3d(dets, nms_thresh, max_count=64)
# dets(nd-array): [x, y, z, dx, dy, dz, score] mode boxes and score
```

Directly use 3d roi align pool function
```
from layers import roi_align_3d
pooler = roi_align_3d.ROIAlignPooler(output_size, scales)
pooler(x, boxes)
# x (list[Tensor]): feature maps for each level
# boxes (list[bboxes]): boxes to be used to perform the pooling operation, bboxes with columns: z1y1x1z2y2x2.
```

### Function description
#### 2d ops
Support for nms, roi align, deform pool and deform conv.
- nms
```
at::Tensor nms(const at::Tensor boxes, float nms_overlap_thresh);
// boxes columns: x1, y1, x2, y2, s
```

- roi_align_forward
```
at::Tensor roi_align_forward(const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int sampling_ratio);
// input: [n, c, h, w]
// rois columns: bi, x1, y1, x2, y2
```

- roi_align_backward
```
at::Tensor roi_align_backward(const at::Tensor& grad,
                                  const at::Tensor& rois,
                                  const float spatial_scale,
                                  const int pooled_height,
                                  const int pooled_width,
                                  const int batch_size,
                                  const int channels,
                                  const int height,
                                  const int width,
                                  const int sampling_ratio);
```

#### 3d ops
Support for nms, roi align.
- nms
```
at::Tensor nms(const at::Tensor boxes, float nms_overlap_thresh);
// boxes columns: x1, y1, x2, y2, z1, z2, s
```

- roi_align_forward
```
at::Tensor roi_align_forward(const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int pooled_depth,
                                 const int sampling_ratio);
// input: [n, c, h, w, d]
// rois columns: bi, x1, y1, x2, y2, z1, z2
```

### References
ROI, NMS: https://github.com/facebookresearch/maskrcnn-benchmark/tree/master/maskrcnn_benchmark/csrc