#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename T>
__device__ T bilinear_interpolate(const T* bottom_data,
    const int height, const int width, const int depth,
    T y, T x, T z,
    const int index /* index for debug only*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width || z < -1.0 || z > depth) {
    //empty
    return 0;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;
  if (z <= 0) z = 0;

  int y_low = (int) y;
  int x_low = (int) x;
  int y_high;
  int x_high;
  int z_low = (int)z;
  int z_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T) x_low;
  } else {
    x_high = x_low + 1;
  }

  if (z_low >= depth - 1) {
    z_high = z_low = depth - 1;
    z = (T)z_low;
  } else {
    z_high = z_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T lz = z - z_low;
  T hy = 1. - ly, hx = 1. - lx, hz = 1. - lz;
  // do bilinear interpolation
  T v1 = bottom_data[y_low * width * depth + x_low * depth + z_low];
  T v2 = bottom_data[y_low * width * depth + x_low * depth + z_high];
  T v3 = bottom_data[y_low * width * depth + x_high * depth + z_low];
  T v4 = bottom_data[y_low * width * depth + x_high * depth + z_high];
  T v5 = bottom_data[y_high * width * depth + x_low * depth + z_low];
  T v6 = bottom_data[y_high * width * depth + x_low * depth + z_high];
  T v7 = bottom_data[y_high * width * depth + x_high * depth + z_low];
  T v8 = bottom_data[y_high * width * depth + x_high * depth + z_high];

  T w1 = hy * hx * hz, w2 = hy * hx * lz, w3 = hy * lx * hz, w4 = hy * lx * lz;
  T w5 = ly * hx * hz, w6 = ly * hx * lz, w7 = ly * lx * hz, w8 = ly * lx * lz;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4 + w5 * v5 + w6 * v6 + w7 * v7 + w8 * v8);

  return val;
}

template <typename T>
__global__ void RoIAlignForward(const int nthreads, const T* bottom_data,
    const int channels,
    const int height, const int width, const int depth,
    const int pooled_height, const int pooled_width, const int pooled_depth,
    const int sampling_ratio,
    const T* bottom_rois, T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw, pd) is an element in the pooled output
    int pd = index % pooled_depth;
    int pw = (index / pooled_depth) % pooled_width;
    int ph = (index / pooled_width / pooled_depth) % pooled_height;
    int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
    int n = index / pooled_width / pooled_height / pooled_depth / channels;

    const T* offset_bottom_rois = bottom_rois + n * 7;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    T roi_start_h = offset_bottom_rois[1] * height;
    T roi_start_w = offset_bottom_rois[2] * width;
    T roi_end_h = offset_bottom_rois[3] * height;
    T roi_end_w = offset_bottom_rois[4] * width;
    T roi_start_d = offset_bottom_rois[5] * depth;
    T roi_end_d = offset_bottom_rois[6] * depth;

    // Force malformed ROIs to be 1x1
    T roi_width = max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = max(roi_end_h - roi_start_h, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);
    T roi_depth = max(roi_end_d - roi_start_d, (T)1.);
    T bin_size_d = static_cast<T>(roi_depth) / static_cast<T>(pooled_depth);

    const T* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width * depth;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);
    int roi_bin_grid_d = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_depth / pooled_depth);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w * roi_bin_grid_d; // e.g. = 8

    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy ++) // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h + static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix ++)
      {
        const T x = roi_start_w + pw * bin_size_w + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);
        for (int iz = 0; iz < roi_bin_grid_d; iz ++)
        {
          const T z = roi_start_d + pd * bin_size_d + static_cast<T>(ix + .5f) * bin_size_d / static_cast<T>(roi_bin_grid_d);

          T val = bilinear_interpolate(offset_bottom_data, height, width, depth, y, x, z, index);
          output_val += val;
        }
      }
    }
    output_val /= count;

    top_data[index] = output_val;
  }
}


template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height, const int width, const int depth,
    T y, T x, T z,
    T & w1, T & w2, T & w3, T & w4, T & w5, T & w6, T & w7, T & w8,
    int & x_low, int & x_high, int & y_low, int & y_high, int & z_low, int & z_high,
    const int index /* index for debug only*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width || z < -1.0 || z > depth) {
    //empty
    w1 = w2 = w3 = w4 = w5 = w6 = w7 = w8 = 0.;
    x_low = x_high = y_low = y_high = z_low = z_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;
  if (z <= 0) z = 0;

  y_low = (int) y;
  x_low = (int) x;
  z_low = (int) z;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T) x_low;
  } else {
    x_high = x_low + 1;
  }

  if (z_low >= depth - 1) {
    z_high = z_low = depth - 1;
    z = (T) z_low;
  } else {
    z_high = z_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T lz = z - z_low;
  T hy = 1. - ly, hx = 1. - lx, hz = 1. - lz;

  // reference in forward
  // T v1 = bottom_data[y_low * width + x_low];
  // T v2 = bottom_data[y_low * width + x_high];
  // T v3 = bottom_data[y_high * width + x_low];
  // T v4 = bottom_data[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  // w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  w1 = hy * hx * hz, w2 = hy * hx * lz, w3 = hy * lx * hz, w4 = hy * lx * lz;
  w5 = ly * hx * hz, w6 = ly * hx * lz, w7 = ly * lx * hz, w8 = ly * lx * lz;

  return;
}

template <typename T>
__global__ void RoIAlignBackwardFeature(const int nthreads, const T* top_diff,
    const int num_rois,
    const int channels, const int height, const int width, const int depth,
    const int pooled_height, const int pooled_width, const int pooled_depth,
    const int sampling_ratio,
    T* bottom_diff,
    const T* bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw, pd) is an element in the pooled output
    int pd = index % pooled_depth;
    int pw = (index / pooled_depth) % pooled_width;
    int ph = (index / pooled_width / pooled_depth) % pooled_height;
    int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
    int n = index / pooled_width / pooled_height / pooled_depth / channels;

    const T* offset_bottom_rois = bottom_rois + n * 7;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    // T roi_start_h = offset_bottom_rois[1] * height;
    // T roi_start_w = offset_bottom_rois[2] * width;
    // T roi_end_h = offset_bottom_rois[3] * height;
    // T roi_end_w = offset_bottom_rois[4] * width;
    // T roi_start_d = offset_bottom_rois[5] * depth;
    // T roi_end_d = offset_bottom_rois[6] * depth;
    
    T roi_start_h = offset_bottom_rois[1];
    T roi_start_w = offset_bottom_rois[2];
    T roi_end_h = offset_bottom_rois[3];
    T roi_end_w = offset_bottom_rois[4];
    T roi_start_d = offset_bottom_rois[5];
    T roi_end_d = offset_bottom_rois[6];

    // Force malformed ROIs to be 1x1
    T roi_width = max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = max(roi_end_h - roi_start_h, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);
    T roi_depth = max(roi_end_d - roi_start_d, (T)1.);
    T bin_size_d = static_cast<T>(roi_depth) / static_cast<T>(pooled_depth);

    T* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width * depth;

    int top_offset    = (n * channels + c) * pooled_height * pooled_width * pooled_depth;
    const T* offset_top_diff = top_diff + top_offset;
    const T top_diff_this_bin = offset_top_diff[ph * pooled_width * pooled_depth + pw * pooled_depth + pd];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);
    int roi_bin_grid_d = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_depth / pooled_depth);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w * roi_bin_grid_d; // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy ++) // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h + static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix ++)
      {
        const T x = roi_start_w + pw * bin_size_w + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);
        for (int iz = 0; iz < roi_bin_grid_d; iz ++)
        {
          const T z = roi_start_d + pd * bin_size_d + static_cast<T>(ix + .5f) * bin_size_d / static_cast<T>(roi_bin_grid_d);

          T w1, w2, w3, w4, w5, w6, w7, w8;
          int x_low, x_high, y_low, y_high, z_low, z_high;

          bilinear_interpolate_gradient(height, width, depth, y, x, z,
              w1, w2, w3, w4, w5, w6, w7, w8,
              x_low, x_high, y_low, y_high, z_low, z_high,
              index);

          T g1 = top_diff_this_bin * w1 / count;
          T g2 = top_diff_this_bin * w2 / count;
          T g3 = top_diff_this_bin * w3 / count;
          T g4 = top_diff_this_bin * w4 / count;
          T g5 = top_diff_this_bin * w5 / count;
          T g6 = top_diff_this_bin * w6 / count;
          T g7 = top_diff_this_bin * w7 / count;
          T g8 = top_diff_this_bin * w8 / count;

          if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0 && z_low >= 0 && z_high >= 0)
          {
            atomicAdd(offset_bottom_diff + y_low * width * depth + x_low * depth + z_low, static_cast<T>(g1));
            atomicAdd(offset_bottom_diff + y_low * width * depth + x_low * depth + z_high, static_cast<T>(g2));
            atomicAdd(offset_bottom_diff + y_low * width * depth + x_high * depth + z_low, static_cast<T>(g3));
            atomicAdd(offset_bottom_diff + y_low * width * depth + x_high * depth + z_high, static_cast<T>(g4));
            atomicAdd(offset_bottom_diff + y_high * width * depth + x_low * depth + z_low, static_cast<T>(g5));
            atomicAdd(offset_bottom_diff + y_high * width * depth + x_low * depth + z_high, static_cast<T>(g6));
            atomicAdd(offset_bottom_diff + y_high * width * depth + x_high * depth + z_low, static_cast<T>(g7));
            atomicAdd(offset_bottom_diff + y_high * width * depth + x_high * depth + z_high, static_cast<T>(g8));
          } // if
        } // iz
      } // ix
    } // iy
  } // CUDA_1D_KERNEL_LOOP
} // RoIAlignBackward


at::Tensor ROIAlign_forward_cuda(const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int pooled_depth,
                                 const int sampling_ratio) {
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");
  // input: [n, c, h, w, d]
  // rois columns: y1, x1, y2, x2, z1, z2, s

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);
  auto depth = input.size(4);

  auto output = at::empty({num_rois, channels, pooled_height, pooled_width, pooled_depth}, input.options());
  auto output_size = num_rois * pooled_height * pooled_width * pooled_depth * channels;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv((long)output_size, 512L), 4096L));
  dim3 block(512);

  if (output.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES(input.type(), "ROIAlign_forward", [&] {
    RoIAlignForward<scalar_t><<<grid, block, 0, stream>>>(
         output_size,
         input.contiguous().data<scalar_t>(),
         channels,
         height,
         width,
         depth,
         pooled_height,
         pooled_width,
         pooled_depth,
         sampling_ratio,
         rois.contiguous().data<scalar_t>(),
         output.data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return output;
}

// TODO remove the dependency on input and use instead its sizes -> save memory
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
                                  const int sampling_ratio) {
  AT_ASSERTM(grad.type().is_cuda(), "grad must be a CUDA tensor");
  AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");

  auto num_rois = rois.size(0);
  auto grad_input = at::zeros({batch_size, channels, height, width, depth}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv((long)grad.numel(), 512L), 4096L));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return grad_input;
  }

  AT_DISPATCH_FLOATING_TYPES(grad.type(), "ROIAlign_backward", [&] {
    RoIAlignBackwardFeature<scalar_t><<<grid, block, 0, stream>>>(
         grad.numel(),
         grad.contiguous().data<scalar_t>(),
         num_rois,
         channels,
         height,
         width,
         depth,
         pooled_height,
         pooled_width,
         pooled_depth,
         sampling_ratio,
         grad_input.data<scalar_t>(),
         rois.contiguous().data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return grad_input;
}
