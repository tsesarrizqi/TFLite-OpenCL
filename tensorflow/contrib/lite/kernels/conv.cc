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
#include <unistd.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

//note: android log
#include <android/log.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>

//note: android opencl
#include "CL/cl.h"
// #include <jni.h>
#include <string.h>
// #include <android/log.h>

// note: timer
#include <time.h>
#include <sys/time.h>

#define MEM_SIZE (128)

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/gemm_support.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/multithreaded_conv.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/quantization_util.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"
#include "tensorflow/contrib/lite/kernels/padding.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace conv {

// This file has three implementation of Conv.
enum KernelType {
  kReference,
  kGenericOptimized,  // Neon-free
  kNeonOptimized,
};

struct OpData {
  // IDs are the arbitrary identifiers used by TF Lite to identify and access
  // memory buffers.
  int im2col_id;
  int hwcn_weights_id;

  TfLitePaddingValues padding;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multipler plus a left shift.
  int32_t output_multiplier;
  int output_shift;
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
  // Indexes are the offset to the memory buffer in the array used to keep track
  // of the allocated temporaries.
  int32_t im2col_index;
  int32_t hwcn_weights_index;
  bool need_hwcn_weights;
  bool have_weights_been_transposed;
  bool need_im2col;
};
    
cl_context context_cl_global = NULL;       
cl_command_queue queue_global = NULL;
cl_program program_global = NULL;

// const char *kernelSource_conv =           "\n" \
// "__kernel void conv(__global float* input_data,   \n" \
// "          __global float* filter_data,   \n" \
// "          __global float* bias_data,   \n" \
// "          __global float* output_data,  \n" \
// "          int stride_width, int stride_height,   \n" \
// "          int pad_width, int pad_height,   \n" \
// "          __global int* dim_sizes, __global int* dim_strides,  \n" \
// "          float output_activation_min, float output_activation_max) {  \n" \
// "  int gid = get_global_id(0);  \n" \
// "  const int batches = dim_sizes[3];  \n" \
// "  const int input_depth = dim_sizes[0];  \n" \
// "  const int output_depth = dim_sizes[7];  \n" \
// "  int batch = gid/output_depth;  \n" \
// "  int out_channel = gid%output_depth;  \n" \
// "  if(gid < batches*output_depth) {  \n" \
// "    const int input_height = dim_sizes[2];  \n" \
// "    const int input_width = dim_sizes[1];  \n" \
// "    const int filter_height = dim_sizes[6];  \n" \
// "    const int filter_width = dim_sizes[5];  \n" \
// "    const int output_height = dim_sizes[14];  \n" \
// "    const int output_width = dim_sizes[13];  \n" \
// "    for (int out_y = 0; out_y < output_height; ++out_y) {  \n" \
// "      for (int out_x = 0; out_x < output_width; ++out_x) {  \n" \
// "        const int in_x_origin = (out_x * stride_width) - pad_width;  \n" \
// "        const int in_y_origin = (out_y * stride_height) - pad_height;  \n" \
// "        float total = 0.f;  \n" \
// "        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {  \n" \
// "          for (int filter_x = 0; filter_x < filter_width; ++filter_x) {  \n" \
// "            for (int in_channel = 0; in_channel < input_depth; ++in_channel) {  \n" \
// "              const int in_x = in_x_origin + filter_x;  \n" \
// "              const int in_y = in_y_origin + filter_y;  \n" \
// "              if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&  \n" \
// "                  (in_y < input_height)) {  \n" \
// "                float input_value = input_data[in_channel*dim_strides[0] + in_x*dim_strides[1] +   \n" \
// "                                                in_y*dim_strides[2] + batch*dim_strides[3]];  \n" \
// "                float filter_value =  \n" \
// "                    filter_data[in_channel*dim_strides[4] + filter_x*dim_strides[5] +  \n" \
// "                                       filter_y*dim_strides[6] + out_channel*dim_strides[7]];  \n" \
// "                total += (input_value * filter_value);  \n" \
// "              }  \n" \
// "            }  \n" \
// "          }  \n" \
// "        }  \n" \
// "        float bias_value = 0.0f;  \n" \
// "        if (bias_data) {  \n" \
// "          bias_value = bias_data[out_channel*dim_strides[8]];  \n" \
// "        }  \n" \
// "        float max = total+bias_value; \n" \
// "        if(max < output_activation_min) max = output_activation_min; \n" \
// "        float min = max; \n" \
// "        if(min > output_activation_max) min = output_activation_max; \n" \
// "        output_data[out_channel*dim_strides[12] + out_x*dim_strides[13] +   \n" \
// "                     out_y*dim_strides[14] + batch*dim_strides[15]] = min; \n" \
// "      }  \n" \
// "    }  \n" \
// "  }  \n" \
// "}  \n" \
// "\n";

const char *kernelSource_transpose =           "\n" \                               
"__kernel void transpose(__global float* input, __global float* output, \n" \
"    int rows, int cols) {          \n" \
"   int gid = get_global_id(0);                                       \n" \
"  \n" \
"   if(gid < rows*cols) {                                                    \n" \
"      int i = gid/cols; \n" \
"      int j = gid%cols;   \n" \
"      const float in_value = input[gid]; \n" \
"      output[j*rows + i] = in_value; \n" \
"   }                                                                 \n" \
"} \n" \                                                                    
"\n";

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

double get_cpu_time(){
    return (double)clock() / CLOCKS_PER_SEC;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to use as scratch space for im2col, and
  // to carry information from Prepare() to Eval().

  auto* data = new OpData;
  context->AddTensors(context, 1, &data->im2col_id);
  context->AddTensors(context, 1, &data->hwcn_weights_id);
  gemm_support::IncrementUsageCounter(context);
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "Convcc", "init conv");

  return data;
}

void* InitOpenCL(TfLiteContext* context, const char* buffer, size_t length,
  cl_context context_cl, cl_command_queue queue, cl_program program) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to use as scratch space for im2col, and
  // to carry information from Prepare() to Eval().
  
  context_cl_global = context_cl;
  program_global = program;
  queue_global = queue;

  // //OpenCL init
  // cl_int err;

  // err = clGetPlatformIDs(1, &cpPlatform, NULL);
 
  // err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

  // context_cl = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

  // queue = clCreateCommandQueue(context_cl, device_id, 0, &err);

  // program = clCreateProgramWithSource(context_cl, 1,
  //                         (const char **) & kernelSource_conv, NULL, &err);

  // clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

  auto* data = new OpData;
  context->AddTensors(context, 1, &data->im2col_id);
  context->AddTensors(context, 1, &data->hwcn_weights_id);
  gemm_support::IncrementUsageCounter(context);
  // //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "Convcc", "init conv");

  // clTest();

  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  // clReleaseProgram(program);
  // clReleaseCommandQueue(queue);
  // clReleaseContext(context_cl);
  gemm_support::DecrementUsageCounter(context);
  delete reinterpret_cast<OpData*>(buffer);
}


// void OpenCLTransposeFloatTensor(const float* input, const int rows, const int cols, float* output) {
//   cl_mem d_a;
//   cl_mem d_b;

//   // cl_platform_id cpPlatform;
//   // cl_device_id device_id;    
//   // cl_context context;       
//   // cl_command_queue queue;   
//   // cl_program program;       
//   cl_kernel kernel;

//   size_t globalSize, localSize;
//   localSize = 32;
//   globalSize = ceil(batches*output_depth/(localSize*1.0))*localSize;

//   cl_int err;

//   // err = clGetPlatformIDs(1, &cpPlatform, NULL);
 
//   // err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

//   // context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

//   // queue = clCreateCommandQueue(context, device_id, 0, &err);

//   // program = clCreateProgramWithSource(context, 1,
//   //                         (const char **) & kernelSource, NULL, &err);

//   // clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

//   kernel = clCreateKernel(program, "transpose", &err);

//   d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, rows*cols*sizeof(float), NULL, NULL);
//   d_b = clCreateBuffer(context, CL_MEM_WRITE_ONLY, rows*cols*sizeof(float), NULL, NULL);

//   err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
//                                  rows*cols*sizeof(float), input, 0, NULL, NULL);

//   err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
//   err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
//   err  = clSetKernelArg(kernel, 2, sizeof(int), &rows);
//   err  = clSetKernelArg(kernel, 3, sizeof(int), &cols);

//   err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

//   clFinish(queue);

//   clEnqueueReadBuffer(queue, d_b, CL_TRUE, 0, rows*cols*sizeof(float), output, 0, NULL, NULL );

//   clReleaseMemObject(d_a);
//   clReleaseMemObject(d_b);
//   clReleaseProgram(program);
//   clReleaseKernel(kernel);
//   clReleaseCommandQueue(queue);
//   clReleaseContext(context);
// }


// Naive implementation of transpose for floats. Could be optimized to be more
// cache friendly, but for now it's a one-time cost on first run, and we would
// prefer to remove the need to do this at all eventually.
void TransposeFloatTensor(TfLiteTensor* input, TfLiteTensor* output) {
  // //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "transpose conv");
  const int rows = output->dims->data[1];
  const int cols = output->dims->data[0];
  const float* input_data = GetTensorData<float>(input);
  float* output_data = GetTensorData<float>(output);

  // Start Timers
  // double wall0 = get_wall_time();
  // double cpu0  = get_cpu_time();

  // //OpenCL
  // OpenCLTransposeFloatTensor(input_data,rows,cols,output_data);

  // Naive
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const float in_value = input_data[i * cols + j];
      output_data[j * rows + i] = in_value;
    }
  }

  // Stop timers
  // double wall1 = get_wall_time();
  // double cpu1  = get_cpu_time();

  // double wall = wall1 - wall0;
  // double cpu = cpu1 - cpu0;

  // // note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "Transposeruntime", "Walltime: %lf", wall);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  bool hasBias = node->inputs->size == 3;
  // Check number of inputs/outputs
  TF_LITE_ENSURE(context, hasBias || node->inputs->size == 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
  TfLiteTensor* input = &context->tensors[node->inputs->data[0]];
  TfLiteTensor* filter = &context->tensors[node->inputs->data[1]];
  // Check dimensionality of input, filter
  TF_LITE_ENSURE_EQ(context, input->dims->size, 4);
  TF_LITE_ENSURE_EQ(context, filter->dims->size, 4);
  // Check input channels matching filter
  TF_LITE_ENSURE_EQ(context, input->dims->data[3], filter->dims->data[3]);

  // Check types. (We assume that UINT8 refers to quantized tensors)
  TfLiteType data_type = input->type;
  TF_LITE_ENSURE(context,
                 data_type == kTfLiteFloat32 || data_type == kTfLiteUInt8);
  TF_LITE_ENSURE_EQ(context, output->type, data_type);
  TF_LITE_ENSURE_EQ(context, filter->type, data_type);

  TfLiteTensor* bias = nullptr;

  // TODO(ahentz): At this point the optimized versions require 'bias'. We can
  // either change that or document that convolution requires it.
  TF_LITE_ENSURE(context, hasBias);

  if (hasBias) {
    bias = &context->tensors[node->inputs->data[2]];
    if (data_type == kTfLiteUInt8) {
      TF_LITE_ENSURE_EQ(context, bias->type, kTfLiteInt32);
      TF_LITE_ENSURE_EQ(context, bias->params.zero_point, 0);
    } else {
      TF_LITE_ENSURE_EQ(context, bias->type, data_type);
    }
    TF_LITE_ENSURE_EQ(context, bias->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, bias->dims->data[0], filter->dims->data[0]);
  }

  int channels_out = filter->dims->data[0];
  int width = input->dims->data[2];
  int height = input->dims->data[1];
  int filter_width = filter->dims->data[2];
  int filter_height = filter->dims->data[1];
  int batches = input->dims->data[0];

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params->padding;
  auto computeOutSize = [padding](int imageSize, int filterSize,
                                  int stride) -> int {
    return padding == kTfLitePaddingSame
               ? (imageSize + stride - 1) / stride
               : padding == kTfLitePaddingValid
                     ? (imageSize - filterSize + stride) / stride
                     : 0;
  };

  int outWidth = computeOutSize(width, filter_width, params->stride_width);
  int outHeight = computeOutSize(height, filter_height, params->stride_height);

  data->padding.height =
      ComputePadding(params->stride_height, height, filter_height, outHeight);
  data->padding.width =
      ComputePadding(params->stride_width, width, filter_width, outWidth);

  TF_LITE_ENSURE(context, hasBias);

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  if (data_type != kTfLiteFloat32) {
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, filter, bias, output, &real_multiplier));
    QuantizeMultiplierSmallerThanOne(real_multiplier, &data->output_multiplier,
                                     &data->output_shift);
    CalculateActivationRangeUint8(params->activation, output,
                                  &data->output_activation_min,
                                  &data->output_activation_max);
  }

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = batches;
  output_size->data[1] = outHeight;
  output_size->data[2] = outWidth;
  output_size->data[3] = channels_out;
  auto output_status = context->ResizeTensor(context, output, output_size);

  if (output_status != kTfLiteOk) return output_status;

  // We don't always need to allocate im2col. It is only used in some versions
  // of the optimized Conv. This test just mimics something that happens inside
  // optimized_ops.h, in order to avoid a DCHECK(!im2col_data).
  data->need_im2col =
      (params->stride_width != 1 || params->stride_height != 1 ||
       filter_width != 1 || filter_height != 1);
  // If we're using the optimized multithreaded EigenTensor implementation of
  // convolution, it expects the filter weights to be transposed compared to
  // the normal TF Lite buffer format. Typical TF Lite weights are
  // [filter_count, filter_height, filter_width, input_depth], but for the float
  // implementation we need them as [filter_height, filter_width, input_depth,
  // filter_count]. We get to that format by transposing, and create a temporary
  // buffer to store the results.
  // This path is only used for float processing, so only create the buffer if
  // we're running with that data type.
  data->need_hwcn_weights = (data_type == kTfLiteFloat32);

  int temporaries_count = 0;
  if (data->need_im2col) {
    data->im2col_index = temporaries_count;
    ++temporaries_count;
  }
  if (data->need_hwcn_weights) {
    data->hwcn_weights_index = temporaries_count;
    ++temporaries_count;
  }

  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(temporaries_count);

  if (data->need_im2col) {
    node->temporaries->data[data->im2col_index] = data->im2col_id;

    TfLiteIntArray* im2col_size = TfLiteIntArrayCreate(4);

    int input_depth = input->dims->data[3];
    im2col_size->data[0] = output_size->data[0];
    im2col_size->data[1] = output_size->data[1];
    im2col_size->data[2] = output_size->data[2];
    im2col_size->data[3] = input_depth * filter_height * filter_width;

    TfLiteTensor* im2col =
        &context->tensors[node->temporaries->data[data->im2col_index]];
    im2col->type = data_type;
    im2col->allocation_type = kTfLiteArenaRw;
    auto im2col_status = context->ResizeTensor(context, im2col, im2col_size);
    if (im2col_status != kTfLiteOk) return im2col_status;
  }

  if (data->need_hwcn_weights) {
    node->temporaries->data[data->hwcn_weights_index] = data->hwcn_weights_id;
    TfLiteIntArray* hwcn_weights_size = TfLiteIntArrayCreate(2);

    // Because we're treating the filter weights as a matrix when we do the
    // transpose, we allocate the buffer with a two-dimensional shape, where one
    // dimension is the number of elements in each filter, and the second is the
    // total number of filters.
    int input_depth = input->dims->data[3];
    hwcn_weights_size->data[0] = (filter_height * filter_width * input_depth);
    hwcn_weights_size->data[1] = channels_out;

    TfLiteTensor* hwcn_weights =
        &context->tensors[node->temporaries->data[data->hwcn_weights_index]];
    hwcn_weights->type = data_type;
    hwcn_weights->allocation_type = kTfLiteDynamic;
    // Make sure we release any previous allocations before we reallocate.
    // TODO(petewarden): Persistent arenas would be a better fit for this, but
    // they aren't fully implemented yet.
    if (hwcn_weights->data.raw) {
      free(hwcn_weights->data.raw);
      hwcn_weights->data.raw = nullptr;
    }
    auto hwcn_weights_status =
        context->ResizeTensor(context, hwcn_weights, hwcn_weights_size);
    if (hwcn_weights_status != kTfLiteOk) return hwcn_weights_status;
    hwcn_weights->data.raw = static_cast<char*>(malloc(hwcn_weights->bytes));

    // TODO(petewarden): If Resize() is called when the size hasn't actually
    // changed, this will do extra redundant work.
    data->have_weights_been_transposed = false;
  }

  return kTfLiteOk;
}

template <KernelType kernel_type>
void EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                   TfLiteConvParams* params, OpData* data, TfLiteTensor* input,
                   TfLiteTensor* filter, TfLiteTensor* bias,
                   TfLiteTensor* im2col, TfLiteTensor* hwcn_weights,
                   TfLiteTensor* output) {
  gemmlowp::GemmContext* gemm_context = gemm_support::GetFromContext(context);

  auto input_offset = -input->params.zero_point;
  auto filter_offset = -filter->params.zero_point;
  auto output_offset = output->params.zero_point;

  if (kernel_type == kReference) {
    reference_ops::Conv(
        GetTensorData<uint8_t>(input), GetTensorDims(input), input_offset,
        GetTensorData<uint8_t>(filter), GetTensorDims(filter), filter_offset,
        GetTensorData<int32_t>(bias), GetTensorDims(bias), params->stride_width,
        params->stride_height, data->padding.width, data->padding.height,
        output_offset, data->output_multiplier, data->output_shift,
        data->output_activation_min, data->output_activation_max,
        GetTensorData<uint8_t>(output), GetTensorDims(output),
        GetTensorData<uint8_t>(im2col), GetTensorDims(im2col), gemm_context);
  } else {
    optimized_ops::Conv(
        GetTensorData<uint8_t>(input), GetTensorDims(input), input_offset,
        GetTensorData<uint8_t>(filter), GetTensorDims(filter), filter_offset,
        GetTensorData<int32_t>(bias), GetTensorDims(bias), params->stride_width,
        params->stride_height, data->padding.width, data->padding.height,
        output_offset, data->output_multiplier, data->output_shift,
        data->output_activation_min, data->output_activation_max,
        GetTensorData<uint8_t>(output), GetTensorDims(output),
        GetTensorData<uint8_t>(im2col), GetTensorDims(im2col), gemm_context);
  }
}

template <KernelType kernel_type>
void EvalFloat(TfLiteContext* context, TfLiteNode* node,
               TfLiteConvParams* params, OpData* data, TfLiteTensor* input,
               TfLiteTensor* filter, TfLiteTensor* bias, TfLiteTensor* im2col,
               TfLiteTensor* hwcn_weights, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRangeFloat(params->activation, &output_activation_min,
                                &output_activation_max);

  if (kernel_type == kReference) {
    __android_log_print(ANDROID_LOG_INFO, "Convcc", "reference conv");
    reference_ops::Conv(GetTensorData<float>(input), GetTensorDims(input),
                        GetTensorData<float>(filter), GetTensorDims(filter),
                        GetTensorData<float>(bias), GetTensorDims(bias),
                        params->stride_width, params->stride_height,
                        data->padding.width, data->padding.height,
                        output_activation_min, output_activation_max,
                        GetTensorData<float>(output), GetTensorDims(output),
                        GetTensorData<float>(im2col), GetTensorDims(im2col));
  } else {
    const float* filter_data;
    if (data->need_hwcn_weights) {
      filter_data = GetTensorData<float>(hwcn_weights);
    } else {
      filter_data = GetTensorData<float>(filter);
    }
    //note: andoird log
    // __android_log_print(ANDROID_LOG_INFO, "Convcc", "multithread conv");
    multithreaded_ops::Conv(
        GetTensorData<float>(input), GetTensorDims(input), filter_data,
        GetTensorDims(filter), GetTensorData<float>(bias), GetTensorDims(bias),
        params->stride_width, params->stride_height, data->padding.width,
        data->padding.height, params->padding, output_activation_min,
        output_activation_max, GetTensorData<float>(output),
        GetTensorDims(output), GetTensorData<float>(im2col),
        GetTensorDims(im2col));
    // multithreaded_ops::ConvOpenCL(
    //     GetTensorData<float>(input), GetTensorDims(input), filter_data,
    //     GetTensorDims(filter), GetTensorData<float>(bias), GetTensorDims(bias),
    //     params->stride_width, params->stride_height, data->padding.width,
    //     data->padding.height, params->padding, output_activation_min,
    //     output_activation_max, GetTensorData<float>(output),
    //     GetTensorDims(output), GetTensorData<float>(im2col),
    //     GetTensorDims(im2col),
    //     context_cl_global, queue_global, program_global);
    
  }
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
  TfLiteTensor* input = &context->tensors[node->inputs->data[0]];
  TfLiteTensor* filter = &context->tensors[node->inputs->data[1]];
  bool hasBias = node->inputs->size == 3;
  TfLiteTensor* bias =
      hasBias ? &context->tensors[node->inputs->data[2]] : nullptr;
  TfLiteTensor* im2col =
      data->need_im2col
          ? &context->tensors[node->temporaries->data[data->im2col_index]]
          : nullptr;
  TfLiteTensor* hwcn_weights =
      data->need_hwcn_weights
          ? &context->tensors[node->temporaries->data[data->hwcn_weights_index]]
          : nullptr;

  if (data->need_hwcn_weights && !data->have_weights_been_transposed) {
    TransposeFloatTensor(filter, hwcn_weights);
    data->have_weights_been_transposed = true;
  }

  // TODO(aselle): Consider whether float conv and quantized conv should be
  // separate ops to avoid dispatch overhead here.
  switch (input->type) {  // Already know in/outtypes are same.
    case kTfLiteFloat32:
      EvalFloat<kernel_type>(context, node, params, data, input, filter, bias,
                             im2col, hwcn_weights, output);
      break;
    case kTfLiteUInt8:
      EvalQuantized<kernel_type>(context, node, params, data, input, filter,
                                 bias, im2col, hwcn_weights, output);
      break;
    default:
      context->ReportError(context, "Type not currently supported.");
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace conv

TfLiteRegistration* Register_CONVOLUTION_REF() {
  static TfLiteRegistration r = {conv::Init, conv::Free, conv::Prepare,
                                 conv::Eval<conv::kReference>, conv::InitOpenCL};
  return &r;
}

TfLiteRegistration* Register_CONVOLUTION_GENERIC_OPT() {
  static TfLiteRegistration r = {conv::Init, conv::Free, conv::Prepare,
                                 conv::Eval<conv::kGenericOptimized>, conv::InitOpenCL};
  return &r;
}

TfLiteRegistration* Register_CONVOLUTION_NEON_OPT() {
  static TfLiteRegistration r = {conv::Init, conv::Free, conv::Prepare,
                                 conv::Eval<conv::kNeonOptimized>, conv::InitOpenCL};
  return &r;
}

TfLiteRegistration* Register_CONV_2D() {
#ifdef USE_NEON
  return Register_CONVOLUTION_NEON_OPT();
#else
  return Register_CONVOLUTION_GENERIC_OPT();
#endif
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
