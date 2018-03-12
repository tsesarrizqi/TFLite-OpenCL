/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_MULTITHREAD_CONV
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_MULTITHREAD_CONV

#include <assert.h>
#include <stdint.h>
#include <sys/types.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>

//note: android opencl
#include "../CL/cl.h"

//note: android log
#include <android/log.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>

//note: string
#include <string>
#include <iostream>

// note: timer
#include <time.h>
#include <sys/time.h>

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/kernels/internal/common.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/eigen_spatial_convolutions.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"

namespace tflite {
namespace multithreaded_ops {

const char *kernelSource =           "\n" \
"__kernel void conv(__global float* input_data,   \n" \
"          __global float* filter_data,   \n" \
"          __global float* bias_data,   \n" \
"          __global float* output_data,  \n" \
"          int stride_width, int stride_height,   \n" \
"          int pad_width, int pad_height,   \n" \
"          __global int* dim_sizes, __global int* dim_strides,  \n" \
"          float output_activation_min, float output_activation_max) {  \n" \
"  int gid = get_global_id(0);  \n" \
"  const int batches = dim_sizes[3];  \n" \
"  const int input_depth = dim_sizes[0];  \n" \
"  const int output_depth = dim_sizes[7];  \n" \
"  int batch = gid/output_depth;  \n" \
"  int out_channel = gid%output_depth;  \n" \
"  if(gid < batches*output_depth) {  \n" \
"    const int input_height = dim_sizes[2];  \n" \
"    const int input_width = dim_sizes[1];  \n" \
"    const int filter_height = dim_sizes[6];  \n" \
"    const int filter_width = dim_sizes[5];  \n" \
"    const int output_height = dim_sizes[14];  \n" \
"    const int output_width = dim_sizes[13];  \n" \
"    for (int out_y = 0; out_y < output_height; ++out_y) {  \n" \
"      for (int out_x = 0; out_x < output_width; ++out_x) {  \n" \
"        const int in_x_origin = (out_x * stride_width) - pad_width;  \n" \
"        const int in_y_origin = (out_y * stride_height) - pad_height;  \n" \
"        float total = 0.f;  \n" \
"        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {  \n" \
"          for (int filter_x = 0; filter_x < filter_width; ++filter_x) {  \n" \
"            for (int in_channel = 0; in_channel < input_depth; ++in_channel) {  \n" \
"              const int in_x = in_x_origin + filter_x;  \n" \
"              const int in_y = in_y_origin + filter_y;  \n" \
"              if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&  \n" \
"                  (in_y < input_height)) {  \n" \
"                float input_value = input_data[in_channel*dim_strides[0] + in_x*dim_strides[1] +   \n" \
"                                                in_y*dim_strides[2] + batch*dim_strides[3]];  \n" \
"                float filter_value =  \n" \
"                    filter_data[in_channel*dim_strides[4] + filter_x*dim_strides[5] +  \n" \
"                                       filter_y*dim_strides[6] + out_channel*dim_strides[7]];  \n" \
"                total += (input_value * filter_value);  \n" \
"              }  \n" \
"            }  \n" \
"          }  \n" \
"        }  \n" \
"        float bias_value = 0.0f;  \n" \
"        if (bias_data) {  \n" \
"          bias_value = bias_data[out_channel*dim_strides[8]];  \n" \
"        }  \n" \
"        float max = total+bias_value; \n" \
"        if(max < output_activation_min) max = output_activation_min; \n" \
"        float min = max; \n" \
"        if(min > output_activation_max) min = output_activation_max; \n" \
"        output_data[out_channel*dim_strides[12] + out_x*dim_strides[13] +   \n" \
"                     out_y*dim_strides[14] + batch*dim_strides[15]] = min; \n" \
"      }  \n" \
"    }  \n" \
"  }  \n" \
"}  \n" \
"\n";


class EigenThreadPoolWrapper : public Eigen::ThreadPoolInterface {
 public:
  explicit EigenThreadPoolWrapper(Eigen::ThreadPool* pool) : pool_(pool) {}
  ~EigenThreadPoolWrapper() override {}

  void Schedule(std::function<void()> fn) override {
    pool_->Schedule(std::move(fn));
  }
  int NumThreads() const override { return pool_->NumThreads(); }
  int CurrentThreadId() const override { return pool_->CurrentThreadId(); }

 private:
  Eigen::ThreadPool* pool_ = nullptr;
};

// We have a single global threadpool for all convolution operations. This means
// that inferences started from different threads may block each other, but
// since the underlying resource of CPU cores should be consumed by the
// operations anyway, it shouldn't affect overall performance.
const Eigen::ThreadPoolDevice& GetThreadPoolDevice() {
  const int thread_count = 4;
  static Eigen::ThreadPool* tp = new Eigen::ThreadPool(thread_count);
  static EigenThreadPoolWrapper* thread_pool_wrapper =
      new EigenThreadPoolWrapper(tp);
  static Eigen::ThreadPoolDevice* device =
      new Eigen::ThreadPoolDevice(thread_pool_wrapper, thread_count);
  return *device;
}

// Shorthands for the types we need when interfacing with the EigenTensor
// library.
typedef Eigen::TensorMap<
    Eigen::Tensor<float, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
    EigenMatrix;
typedef Eigen::TensorMap<
    Eigen::Tensor<const float, 2, Eigen::RowMajor, Eigen::DenseIndex>,
    Eigen::Aligned>
    ConstEigenMatrix;

typedef Eigen::TensorMap<
    Eigen::Tensor<float, 4, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
    EigenTensor;
typedef Eigen::TensorMap<
    Eigen::Tensor<const float, 4, Eigen::RowMajor, Eigen::DenseIndex>,
    Eigen::Aligned>
    ConstEigenTensor;

// Utility functions we need for the EigenTensor API.
template <typename Device, typename T>
struct MatMulConvFunctor {
  // Computes on device "d": out = in0 * in1, where * is matrix
  // multiplication.
  void operator()(
      const Device& d, EigenMatrix out, ConstEigenMatrix in0,
      ConstEigenMatrix in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair) {
    out.device(d) = in0.contract(in1, dim_pair);
  }
};

template <class T>
class EigenTensorConvFunctor {
 private:
  Eigen::PaddingType TfLitePadding2EigenPadding(TfLitePadding padding) {
    switch (padding) {
      case kTfLitePaddingValid:
        return Eigen::PADDING_VALID;
      case kTfLitePaddingSame:
        return Eigen::PADDING_SAME;
      case kTfLitePaddingUnknown:
        assert(false);  // should never get here.
        return Eigen::PADDING_VALID;
    }
    return Eigen::PADDING_SAME;  // Prevent compiler warning about missing
                                 // return
  }

 public:
  void operator()(const T* input_data, T* im2col_buffer, int input_batches,
                  int input_height, int input_width, int input_depth,
                  const T* filter_data, int filter_height, int filter_width,
                  int filter_count, int stride_rows, int stride_cols,
                  int pad_width, int pad_height, TfLitePadding padding,
                  T* output_data, int output_height, int output_width) {
    // char* type_input_data = ;
    // char* type_filter_data = typeid(filter_data).name();
    // char* type_output_data = typeid(output_data).name();

    //note: andoird log
    // __android_log_print(ANDROID_LOG_INFO, "multithread_conv_var1", "input_data: %s\nfilter_data: %s\noutput_data: %s\n",
    //   typeid(input_data).name(),typeid(filter_data).name(),typeid(output_data).name());
    // __android_log_print(ANDROID_LOG_INFO, "multithread_conv_var2", "pad1: %d\npad2: %d\n",pad_width,pad_height);

    const Eigen::ThreadPoolDevice& device = GetThreadPoolDevice();

    const bool is_1x1_kernel = (filter_height == 1 && filter_width == 1 &&
                                stride_rows == 1 && stride_cols == 1);
    if (is_1x1_kernel) {
      // For 1x1 kernel, the 2D convolution is reduced to matrix
      // multiplication.
      const int conv_width = output_height * output_width;
      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      EigenMatrix output(output_data, conv_width, filter_count);
      ConstEigenMatrix input(input_data, conv_width, input_depth);
      ConstEigenMatrix filter(filter_data, input_depth, filter_count);
      //note: andoird log
      // __android_log_print(ANDROID_LOG_INFO, "multithread_conv", "1x1 kernel");
      MatMulConvFunctor<Eigen::ThreadPoolDevice, T>()(device, output, input,
                                                      filter, dim_pair);
    } else if (filter_height == input_height && filter_width == input_width &&
               pad_width == 0 && pad_height == 0) {
      // If the input data and filter have the same height/width,
      // the 2D convolution is reduced to matrix multiplication.
      const int k =  // Length of reduction dimension.
          filter_width * filter_height * input_depth;
      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      EigenMatrix output(output_data, 1, filter_count);
      ConstEigenMatrix input(input_data, 1, k);
      ConstEigenMatrix filter(filter_data, k, filter_count);
      //note: andoird log
      // __android_log_print(ANDROID_LOG_INFO, "multithread_conv", "input=filter height width");
      MatMulConvFunctor<Eigen::ThreadPoolDevice, T>()(device, output, input,
                                                      filter, dim_pair);
    } else {
      EigenTensor output(output_data, input_batches, output_height,
                         output_width, filter_count);
      ConstEigenTensor input(input_data, input_batches, input_height,
                             input_width, input_depth);
      ConstEigenTensor filter(filter_data, filter_height, filter_width,
                              input_depth, filter_count);
      //note: andoird log
      // __android_log_print(ANDROID_LOG_INFO, "multithread_conv", "spatial conv");
      output.device(device) =
          Eigen::SpatialConvolution(input, filter, stride_cols, stride_rows,
                                    TfLitePadding2EigenPadding(padding));
    }
  }
};

inline double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

inline double get_cpu_time(){
    return (double)clock() / CLOCKS_PER_SEC;
}

inline void OpenCLConv(const float* input_data, const int input_size,
          const float* filter_data, const int filter_size,
          const float* bias_data, const int bias_size,
          float* output_data, const int output_size,
          int stride_width, int stride_height, 
          int pad_width, int pad_height, 
          const int* dim_sizes, const int* dim_strides,
          float output_activation_min, float output_activation_max,
          cl_context context, cl_command_queue queue, cl_program program) {
  cl_mem d_input;
  cl_mem d_filter;
  cl_mem d_bias;
  cl_mem d_output;
  cl_mem d_dim_sizes;
  cl_mem d_dim_strides;

  // cl_platform_id cpPlatform;
  // cl_device_id device_id;    
  // cl_context context;       
  // cl_command_queue queue;   
  // cl_program program;       
  cl_kernel kernel;

  size_t globalSize0, globalSize1, localSize;
  localSize = 32;
  
  int batches = dim_sizes[3];
  int output_depth = dim_sizes[7];
  int output_height = dim_sizes[14];  
  int output_width = dim_sizes[13];

  globalSize0 = ceil(batches*output_depth/(localSize*1.0))*localSize;
  globalSize1 = ceil(output_height*output_width/(localSize*1.0))*localSize;

  const size_t local[2] = { localSize, localSize };
  const size_t global[2] = { globalSize0, globalSize1 };

  cl_int err;

  // err = clGetPlatformIDs(1, &cpPlatform, NULL);
 
  // err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

  // context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

  // queue = clCreateCommandQueue(context, device_id, 0, &err);

  // program = clCreateProgramWithSource(context, 1,
  //                         (const char **) & kernelSource, NULL, &err);

  // clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

  // Start Timers
  // double wall0 = get_wall_time();
  // double cpu0  = get_cpu_time();

  kernel = clCreateKernel(program, "conv", &err);

  d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, input_size*sizeof(float), NULL, NULL);
  d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY, filter_size*sizeof(float), NULL, NULL);
  d_bias = clCreateBuffer(context, CL_MEM_READ_ONLY, bias_size*sizeof(float), NULL, NULL);
  d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size*sizeof(float), NULL, NULL);
  d_dim_sizes = clCreateBuffer(context, CL_MEM_READ_ONLY, 16*sizeof(int), NULL, NULL);
  d_dim_strides = clCreateBuffer(context, CL_MEM_READ_ONLY, 16*sizeof(int), NULL, NULL);

  // Stop timers
  // double wall1 = get_wall_time();
  // double cpu1  = get_cpu_time();

  // double wall = wall1 - wall0;
  // double cpu = cpu1 - cpu0;

  // note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime createbuffer: %lf", wall);

  // wall0 = get_wall_time();
  // cpu0  = get_cpu_time();

  err = clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0,
                                 input_size*sizeof(float), input_data, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue, d_filter, CL_TRUE, 0,
                                 filter_size*sizeof(float), filter_data, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue, d_bias, CL_TRUE, 0,
                                 bias_size*sizeof(float), bias_data, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue, d_dim_sizes, CL_TRUE, 0,
                                 16*sizeof(int), dim_sizes, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue, d_dim_strides, CL_TRUE, 0,
                                 16*sizeof(int), dim_strides, 0, NULL, NULL);
  // clFinish(queue);

  // Stop timers
  // wall1 = get_wall_time();
  // cpu1  = get_cpu_time();

  // wall = wall1 - wall0;
  // cpu = cpu1 - cpu0;

  // note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime enqueuebuffer: %lf", wall);

  // wall0 = get_wall_time();
  // cpu0  = get_cpu_time();

  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
  err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_filter);
  err  = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_bias);
  err  = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_output);
  err  = clSetKernelArg(kernel, 4, sizeof(int), &stride_width);
  err  = clSetKernelArg(kernel, 5, sizeof(int), &stride_height);
  err  = clSetKernelArg(kernel, 6, sizeof(int), &pad_width);
  err  = clSetKernelArg(kernel, 7, sizeof(int), &pad_height);
  err  = clSetKernelArg(kernel, 8, sizeof(cl_mem), &d_dim_sizes);
  err  = clSetKernelArg(kernel, 9, sizeof(cl_mem), &d_dim_strides);
  err  = clSetKernelArg(kernel, 10, sizeof(float), &output_activation_min);
  err  = clSetKernelArg(kernel, 11, sizeof(float), &output_activation_max);

  // Stop timers
  // wall1 = get_wall_time();
  // cpu1  = get_cpu_time();

  // wall = wall1 - wall0;
  // cpu = cpu1 - cpu0;

  // note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime setkernelargs: %lf", wall);

  // wall0 = get_wall_time();
  // cpu0  = get_cpu_time();

  err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);

  clFinish(queue);

  // Stop timers
  // wall1 = get_wall_time();
  // cpu1  = get_cpu_time();

  // wall = wall1 - wall0;
  // cpu = cpu1 - cpu0;

  // note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime runkernel: %lf", wall);

  // wall0 = get_wall_time();
  // cpu0  = get_cpu_time();

  clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, output_size*sizeof(float), output_data, 0, NULL, NULL );

  // Stop timers
  // wall1 = get_wall_time();
  // cpu1  = get_cpu_time();

  // wall = wall1 - wall0;
  // cpu = cpu1 - cpu0;

  // note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime readbuffer: %lf", wall);

  // wall0 = get_wall_time();
  // cpu0  = get_cpu_time();

  clReleaseMemObject(d_input);
  clReleaseMemObject(d_filter);
  clReleaseMemObject(d_bias);
  clReleaseMemObject(d_output);
  clReleaseMemObject(d_dim_sizes);
  clReleaseMemObject(d_dim_strides);
  // clReleaseProgram(program);
  clReleaseKernel(kernel);
  // clReleaseCommandQueue(queue);
  // clReleaseContext(context);

  // Stop timers
  // wall1 = get_wall_time();
  // cpu1  = get_cpu_time();

  // wall = wall1 - wall0;
  // cpu = cpu1 - cpu0;

  // note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime cleaning: %lf", wall);
}

inline void Conv(const float* input_data, const Dims<4>& input_dims,
                 const float* filter_data, const Dims<4>& filter_dims,
                 const float* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, TfLitePadding padding,
                 float output_activation_min, float output_activation_max,
                 float* output_data, const Dims<4>& output_dims,
                 float* im2col_data, const Dims<4>& im2col_dims) {
  // // Start Timers
  // double wall0 = get_wall_time();
  // double cpu0  = get_cpu_time();

  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int input_depth = MatchingArraySize(input_dims, 0, filter_dims, 0);
  const int output_depth = MatchingArraySize(filter_dims, 3, output_dims, 0);
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int filter_height = ArraySize(filter_dims, 2);
  const int filter_width = ArraySize(filter_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);

  EigenTensorConvFunctor<float> conv_functor;
  conv_functor(input_data, im2col_data, batches, input_height, input_width,
               input_depth, filter_data, filter_height, filter_width,
               output_depth, stride_height, stride_width, pad_height, pad_width,
               padding, output_data, output_height, output_width);

  optimized_ops::AddBiasAndEvalActivationFunction(
      bias_data, bias_dims, output_data, output_dims, output_activation_min,
      output_activation_max);
}

inline void Conv2(const float* input_data, const Dims<4>& input_dims,
                 const float* filter_data, const Dims<4>& filter_dims,
                 const float* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, float output_activation_min,
                 float output_activation_max, float* output_data,
                 const Dims<4>& output_dims, float* im2col_data,
                 const Dims<4>& im2col_dims) {
  (void)im2col_data;  // only used in optimized code.
  (void)im2col_dims;  // only used in optimized code.
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int input_depth = MatchingArraySize(input_dims, 0, filter_dims, 0);
  const int output_depth = MatchingArraySize(filter_dims, 3, output_dims, 0);
  if (bias_data) {
    TFLITE_DCHECK_EQ(ArraySize(filter_dims, 3), ArraySize(bias_dims, 0));
  }
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int filter_height = ArraySize(filter_dims, 2);
  const int filter_width = ArraySize(filter_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          float total = 0.f;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                const int in_x = in_x_origin + filter_x;
                const int in_y = in_y_origin + filter_y;
                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height)) {
                  float input_value = input_data[Offset(input_dims, in_channel,
                                                        in_x, in_y, batch)];
                  float filter_value =
                      filter_data[Offset(filter_dims, in_channel, filter_x,
                                         filter_y, out_channel)];
                  total += (input_value * filter_value);
                }
              }
            }
          }
          float bias_value = 0.0f;
          if (bias_data) {
            bias_value = bias_data[Offset(bias_dims, out_channel, 0, 0, 0)];
          }
          output_data[Offset(output_dims, out_channel, out_x, out_y, batch)] =
              ActivationFunctionWithMinMax(total + bias_value,
                                           output_activation_min,
                                           output_activation_max);
        }
      }
    }
  }
}

inline void ConvOpenCL(const float* input_data, const Dims<4>& input_dims,
                 const float* filter_data, const Dims<4>& filter_dims,
                 const float* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, TfLitePadding padding,
                 float output_activation_min, float output_activation_max,
                 float* output_data, const Dims<4>& output_dims,
                 float* im2col_data, const Dims<4>& im2col_dims,
                 cl_context context_cl, cl_command_queue queue, cl_program program) {
  // Start Timers
  double wall0 = get_wall_time();
  double cpu0  = get_cpu_time();

  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int input_depth = MatchingArraySize(input_dims, 0, filter_dims, 0);
  const int output_depth = MatchingArraySize(filter_dims, 3, output_dims, 0);
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int filter_height = ArraySize(filter_dims, 2);
  const int filter_width = ArraySize(filter_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);

  int* sizes;
  int* strides;

  sizes = (int*)malloc(16*sizeof(int));
  strides = (int*)malloc(16*sizeof(int));

  //input
  sizes[0] = input_dims.sizes[0];
  sizes[1] = input_dims.sizes[1];
  sizes[2] = input_dims.sizes[2];
  sizes[3] = input_dims.sizes[3];
  strides[0] = input_dims.strides[0];
  strides[1] = input_dims.strides[1];
  strides[2] = input_dims.strides[2];
  strides[3] = input_dims.strides[3];

  //filter
  sizes[4] = filter_dims.sizes[0];
  sizes[5] = filter_dims.sizes[1];
  sizes[6] = filter_dims.sizes[2];
  sizes[7] = filter_dims.sizes[3];
  strides[4] = filter_dims.strides[0];
  strides[5] = filter_dims.strides[1];
  strides[6] = filter_dims.strides[2];
  strides[7] = filter_dims.strides[3];

  //bias
  sizes[8] = bias_dims.sizes[0];
  sizes[9] = bias_dims.sizes[1];
  sizes[10] = bias_dims.sizes[2];
  sizes[11] = bias_dims.sizes[3];
  strides[8] = bias_dims.strides[0];
  strides[9] = bias_dims.strides[1];
  strides[10] = bias_dims.strides[2];
  strides[11] = bias_dims.strides[3];

  //output
  sizes[12] = output_dims.sizes[0];
  sizes[13] = output_dims.sizes[1];
  sizes[14] = output_dims.sizes[2];
  sizes[15] = output_dims.sizes[3];
  strides[12] = output_dims.strides[0];
  strides[13] = output_dims.strides[1];
  strides[14] = output_dims.strides[2];
  strides[15] = output_dims.strides[3];

  int input_size = batches*input_width*input_height*input_depth;
  int filter_size = input_depth*output_depth*filter_width*filter_height;
  int bias_size = output_depth;
  int output_size = batches*output_width*output_height*output_depth;

  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime: %lf", wall);

  OpenCLConv(input_data, input_size,
          filter_data, filter_size,
          bias_data, bias_size,
          output_data, output_size,
          stride_width, stride_height, 
          pad_width, pad_height, 
          sizes, strides,
          output_activation_min, output_activation_max,
          context_cl, queue, program);

  // Conv2(input_data, input_dims,
  //                filter_data, filter_dims,
  //                bias_data, bias_dims,
  //                stride_width, stride_height, pad_width,
  //                pad_height, output_activation_min,
  //                output_activation_max, output_data,
  //                output_dims, im2col_data,
  //                im2col_dims);

  // Stop timers
  double wall1 = get_wall_time();
  double cpu1  = get_cpu_time();

  double wall = wall1 - wall0;
  double cpu = cpu1 - cpu0;

  // note: andoird log
  __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime: %lf", wall);
}

}  // namespace multithreaded_ops
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_MULTITHREAD_CONV
