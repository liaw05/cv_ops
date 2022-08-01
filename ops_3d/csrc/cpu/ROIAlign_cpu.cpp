#include "cpu/vision.h"

// implementation taken from Caffe2
template <typename T>
struct PreCalc {
  int pos1;
  int pos2;
  int pos3;
  int pos4;
  int pos5;
  int pos6;
  int pos7;
  int pos8;
  T w1;
  T w2;
  T w3;
  T w4;
  T w5;
  T w6;
  T w7;
  T w8;
};

template <typename T>
void pre_calc_for_bilinear_interpolate(
    const int height,
    const int width,
    const int depth,
    const int pooled_height,
    const int pooled_width,
    const int pooled_depth,
    const int iy_upper,
    const int ix_upper,
    const int iz_upper,
    T roi_start_h,
    T roi_start_w,
    T roi_start_d,
    T bin_size_h,
    T bin_size_w,
    T bin_size_d,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    int roi_bin_grid_d,
    std::vector<PreCalc<T>>& pre_calc) {
  int pre_calc_index = 0;
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      for (int pd = 0; pd < pooled_depth; pd++) {
        for (int iy = 0; iy < iy_upper; iy++) {
          const T yy = roi_start_h + ph * bin_size_h +
              static_cast<T>(iy + .5f) * bin_size_h /
                  static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
          for (int ix = 0; ix < ix_upper; ix++) {
            const T xx = roi_start_w + pw * bin_size_w +
                static_cast<T>(ix + .5f) * bin_size_w /
                    static_cast<T>(roi_bin_grid_w);
            for (int iz = 0; iz < iz_upper; iz++) {
              const T zz = roi_start_d + pd * bin_size_d +
                  static_cast<T>(iz + .5f) * bin_size_d /
                      static_cast<T>(roi_bin_grid_d);

              T x = xx;
              T y = yy;
              T z = zz;
              // deal with: inverse elements are out of feature map boundary
              if (y < -1.0 || y > height || x < -1.0 || x > width || z < -1.0 || z > depth) {
                // empty
                PreCalc<T> pc;
                pc.pos1 = 0;
                pc.pos2 = 0;
                pc.pos3 = 0;
                pc.pos4 = 0;
                pc.pos5 = 0;
                pc.pos6 = 0;
                pc.pos7 = 0;
                pc.pos8 = 0;
                pc.w1 = 0;
                pc.w2 = 0;
                pc.w3 = 0;
                pc.w4 = 0;
                pc.w5 = 0;
                pc.w6 = 0;
                pc.w7 = 0;
                pc.w8 = 0;
                pre_calc[pre_calc_index] = pc;
                pre_calc_index += 1;
                continue;
              }

              if (y <= 0) {
                y = 0;
              }
              if (x <= 0) {
                x = 0;
              }
              if (z <= 0) {
                z = 0;
              }

              int y_low = (int)y;
              int x_low = (int)x;
              int y_high;
              int x_high;
              int z_low = (int)z;
              int z_high;

              if (y_low >= height - 1) {
                y_high = y_low = height - 1;
                y = (T)y_low;
              } else {
                y_high = y_low + 1;
              }

              if (x_low >= width - 1) {
                x_high = x_low = width - 1;
                x = (T)x_low;
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
              //T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
              T w1 = hy * hx * hz, w2 = hy * hx * lz, w3 = hy * lx * hz, w4 = hy * lx * lz;
              T w5 = ly * hx * hz, w6 = ly * hx * lz, w7 = ly * lx * hz, w8 = ly * lx * lz;

              // save indeces and weights
              PreCalc<T> pc;
              pc.pos1 = y_low * width * depth + x_low * depth + z_low;
              pc.pos2 = y_low * width * depth + x_low * depth + z_high;
              pc.pos3 = y_low * width * depth + x_high * depth + z_low;
              pc.pos4 = y_low * width * depth + x_high * depth + z_high;
              pc.pos5 = y_high * width * depth + x_low * depth + z_low;
              pc.pos6 = y_high * width * depth + x_low * depth + z_high;
              pc.pos7 = y_high * width * depth + x_high * depth + z_low;
              pc.pos8 = y_high * width * depth + x_high * depth + z_high;
              pc.w1 = w1;
              pc.w2 = w2;
              pc.w3 = w3;
              pc.w4 = w4;
              pc.w5 = w5;
              pc.w6 = w6;
              pc.w7 = w7;
              pc.w8 = w8;
              pre_calc[pre_calc_index] = pc;

              pre_calc_index += 1;
            }
          }
        }
      }
    }
  }
}

template <typename T>
void ROIAlignForward_cpu_kernel(
    const int nthreads,
    const T* bottom_data,
    const int channels,
    const int height,
    const int width,
    const int depth,
    const int pooled_height,
    const int pooled_width,
    const int pooled_depth,
    const int sampling_ratio,
    const T* bottom_rois,
    //int roi_cols,
    T* top_data) {
  //AT_ASSERT(roi_cols == 6 || roi_cols == 7);
  int roi_cols = 7;

  int n_rois = nthreads / channels / pooled_width / pooled_height / pooled_depth;
  // (n, c, ph, pw) is an element in the pooled output
  // can be parallelized using omp
  // #pragma omp parallel for num_threads(32)
  for (int n = 0; n < n_rois; n++) {
    int index_n = n * channels * pooled_width * pooled_height * pooled_depth;

    // roi could have 6 or 7 columns
    const T* offset_bottom_rois = bottom_rois + n * roi_cols;
    int roi_batch_ind = 0;
    if (roi_cols == 7) {
      roi_batch_ind = offset_bottom_rois[0];
      offset_bottom_rois++;
    }

    // Do not using rounding; this implementation detail is critical
    // y1,x1,y2,x2,z1,z2
    // T roi_start_h = offset_bottom_rois[0] * height;
    // T roi_start_w = offset_bottom_rois[1] * width;
    // T roi_end_h = offset_bottom_rois[2] * height;
    // T roi_end_w = offset_bottom_rois[3] * width;
    // T roi_start_d = offset_bottom_rois[4] * depth;
    // T roi_end_d = offset_bottom_rois[5] * depth;
    T roi_start_h = offset_bottom_rois[0];
    T roi_start_w = offset_bottom_rois[1];
    T roi_end_h = offset_bottom_rois[2];
    T roi_end_w = offset_bottom_rois[3];
    T roi_start_d = offset_bottom_rois[4];
    T roi_end_d = offset_bottom_rois[5];

    // Force malformed ROIs to be 1x1
    T roi_width = std::max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = std::max(roi_end_h - roi_start_h, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);
    T roi_depth = std::max(roi_end_d - roi_start_d, (T)1.);
    T bin_size_d = static_cast<T>(roi_depth) / static_cast<T>(pooled_depth);

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);
    int roi_bin_grid_d =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_depth / pooled_depth);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w * roi_bin_grid_d; // e.g. = 4

    // we want to precalculate indeces and weights shared by all chanels,
    // this is the key point of optimiation
    std::vector<PreCalc<T>> pre_calc(
        roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height * roi_bin_grid_d * pooled_depth);
    pre_calc_for_bilinear_interpolate(
        height,
        width,
        depth,
        pooled_height,
        pooled_width,
        pooled_depth,
        roi_bin_grid_h,
        roi_bin_grid_w,
        roi_bin_grid_d,
        roi_start_h,
        roi_start_w,
        roi_start_d,
        bin_size_h,
        bin_size_w,
        bin_size_d,
        roi_bin_grid_h,
        roi_bin_grid_w,
        roi_bin_grid_d,
        pre_calc);

    for (int c = 0; c < channels; c++) {
      int index_n_c = index_n + c * pooled_width * pooled_height * pooled_depth;
      const T* offset_bottom_data =
          bottom_data + (roi_batch_ind * channels + c) * height * width * depth;
      int pre_calc_index = 0;

      for (int ph = 0; ph < pooled_height; ph++) {
        for (int pw = 0; pw < pooled_width; pw++) {
          for (int pd = 0; pd < pooled_depth; pd++) {
            int index = index_n_c + ph * pooled_width * pooled_depth + pw * pooled_depth + pd;
            T output_val = 0.;
            for (int iy = 0; iy < roi_bin_grid_h; iy++) {
              for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                for (int iz = 0; iz < roi_bin_grid_d; iz++) {
                  PreCalc<T> pc = pre_calc[pre_calc_index];
                  output_val += pc.w1 * offset_bottom_data[pc.pos1] +
                      pc.w2 * offset_bottom_data[pc.pos2] +
                      pc.w3 * offset_bottom_data[pc.pos3] +
                      pc.w4 * offset_bottom_data[pc.pos4] +
                      pc.w5 * offset_bottom_data[pc.pos5] +
                      pc.w6 * offset_bottom_data[pc.pos6] +
                      pc.w7 * offset_bottom_data[pc.pos7] +
                      pc.w8 * offset_bottom_data[pc.pos8];

                  pre_calc_index += 1;
                }
              }
            }
            output_val /= count;

            top_data[index] = output_val;
          } //fro pd
        } // for pw
      } // for ph
    } // for c
  } // for n
}

at::Tensor ROIAlign_forward_cpu(const at::Tensor& input,
                                const at::Tensor& rois,
                                const int pooled_height,
                                const int pooled_width,
                                const int pooled_depth,
                                const int sampling_ratio) {
  AT_ASSERTM(!input.type().is_cuda(), "input must be a CPU tensor");
  AT_ASSERTM(!rois.type().is_cuda(), "rois must be a CPU tensor");
  // input: [n, c, h, w, d]
  // rois columns: x1, y1, x2, y2, z1, z2, s

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);
  auto depth = input.size(4);

  auto output = at::empty({num_rois, channels, pooled_height, pooled_width, pooled_depth}, input.options());
  auto output_size = num_rois * pooled_height * pooled_width * pooled_depth * channels;

  if (output.numel() == 0) {
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES(input.type(), "ROIAlign_forward", [&] {
    ROIAlignForward_cpu_kernel<scalar_t>(
         output_size,
         input.data<scalar_t>(),
         channels,
         height,
         width,
         depth,
         pooled_height,
         pooled_width,
         pooled_depth,
         sampling_ratio,
         rois.data<scalar_t>(),
         output.data<scalar_t>());
  });
  return output;
}
