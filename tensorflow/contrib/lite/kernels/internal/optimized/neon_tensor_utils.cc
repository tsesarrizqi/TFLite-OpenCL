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
#include <string.h>

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/kernels/internal/common.h"
#include "tensorflow/contrib/lite/kernels/activation_functor.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/tensor_utils_impl.h"

//note: android log
#include <android/log.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>

//note: android opencl
#include "../CL/cl.h"

// note: timer
#include <time.h>
#include <sys/time.h>

#ifdef USE_NEON

#define kFloatWeightsPerNeonLane 4

// note: opencl
// const char *kernelSource =           "\n" \                               
// "__kernel void matrixVectorMul(__global float* resultVector,     \n" \
// "    __global float* matrixA,     \n" \
// "    __global float* vectorB,      \n" \
// "    int width_A,     \n" \
// "    int height_A,     \n" \
// "    int width_B)     \n" \
// "{     \n" \
// "    int idx = get_global_id(0);      \n" \
// "      \n" \
// "    if(idx < height_A) {  \n" \
// "      for(int i = 0; i < width_B; i++) {     \n" \
// "        float value = 0;     \n" \
// "        for (int k = 0; k < width_A; ++k) {     \n" \
// "            value += matrixA[idx * width_A + k] * vectorB[i*width_A+k];     \n" \
// "        }     \n" \
// "        resultVector[i*width_B+idx] = value;     \n" \
// "      }     \n" \
// "    }     \n" \
// "}     \n" \
// "\n";

// const char *kernelSource =           "\n" \
// "__kernel void matrixVectorMul(__global float* C,  \n" \
// "                      const __global float* A,  \n" \
// "                      const __global float* B,  \n" \
// "                      int K, int M, int N) {  \n" \
// "      \n" \
// "    const int row = get_local_id(0); // Local row ID (max: 32)  \n" \
// "    const int col = get_local_id(1); // Local col ID (max: 32)  \n" \
// "    const int globalRow = 32*get_group_id(0) + row; // Row ID of C (0..M)  \n" \
// "    const int globalCol = 32*get_group_id(1) + col; // Col ID of C (0..N)  \n" \
// "   \n" \
// "      __local float Asub[32][32];  \n" \
// "      __local float Bsub[32][32];  \n" \
// "     \n" \
// "      float acc = 0.0;  \n" \
// "        \n" \
// "      const int numTiles = ((K-1)/32)+1;  \n" \
// "      for (int t=0; t<numTiles; t++) {  \n" \
// "     \n" \
// "          const int tiledRow = 32*t + row;  \n" \
// "          const int tiledCol = 32*t + col;  \n" \
// "          if((tiledCol < K) && (globalRow < M)) { \n" \
// "            Asub[col][row] = A[globalRow*K + tiledCol];  \n" \
// "          }   \n" \
// "          else {    \n" \
// "            Asub[col][row] = 0.0;  \n" \
// "          }   \n" \
// "          if((tiledRow < K) && (globalCol < N)) { \n" \
// "            Bsub[col][row] = B[globalCol*K + tiledRow];  \n" \
// "          }   \n" \
// "          else {    \n" \
// "            Bsub[col][row] = 0.0;  \n" \
// "          }   \n" \
// "     \n" \
// "          barrier(CLK_LOCAL_MEM_FENCE);  \n" \
// "     \n" \
// "          for (int k=0; k<32; k++) {  \n" \
// "              acc += Asub[k][row] * Bsub[col][k];  \n" \
// "          }  \n" \
// "     \n" \
// "          barrier(CLK_LOCAL_MEM_FENCE);  \n" \
// "      }  \n" \
// "     \n" \
// "      if((globalRow < M) && (globalCol < N)) {  \n" \
// "          C[globalCol*M + globalRow] = acc;  \n" \
// "      } \n" \
// "} \n" \ 
// "\n";

// const char *kernelSource =           "\n" \
// "__kernel void matrixVectorMul(__global float* C,  \n" \
// "                      const __global float* A,  \n" \
// "                      const __global float* B,  \n" \
// "                      int K, int M, int N) {  \n" \
// "      \n" \
// "    const int row = get_local_id(0); // Local row ID (max: 32)  \n" \
// "    const int col = get_local_id(1); // Local col ID (max: 32)  \n" \
// "    const int globalRow = 32*get_group_id(0) + row; // Row ID of C (0..M)  \n" \
// "    const int globalCol = 32*get_group_id(1) + col; // Col ID of C (0..N)  \n" \
// "   \n" \
// "      __local float Asub[32][32];  \n" \
// "      __local float Bsub[32][32];  \n" \
// "     \n" \
// "      float acc[8];\n" \
// "      for (int w=0; w<8; w++) {\n" \
// "          acc[w] = 0.0f;\n" \
// "      }  \n" \
// "        \n" \
// "      const int numTiles = ((K-1)/32)+1;  \n" \
// "      for (int t=0; t<numTiles; t++) {  \n" \
// "        for (int w=0; w<8; w++) {\n" \
// "          const int tiledRow = 32*t + row;  \n" \
// "          const int tiledCol = 32*t + col;  \n" \
// "          if(((tiledCol+w*4) < K) && (globalRow < M)) { \n" \
// "            Asub[col + w*4][row] = A[globalRow*K + tiledCol + w*4];  \n" \
// "          }   \n" \
// "          else {    \n" \
// "            Asub[col + w*4][row] = 0.0;  \n" \
// "          }   \n" \
// "          if((tiledRow < K) && ((globalCol + w*4) < N)) { \n" \
// "            Bsub[col + w*4][row] = B[(globalCol + w*4)*K + tiledRow];  \n" \
// "          }   \n" \
// "          else {    \n" \
// "            Bsub[col + w*4][row] = 0.0;  \n" \
// "          }   \n" \
// "        }\n" \
// "          barrier(CLK_LOCAL_MEM_FENCE);  \n" \
// "     \n" \
// "          for (int k=0; k<32; k++) {  \n" \
// "            for (int w=0; w<8; w++) {\n" \
// "              acc[w] += Asub[k][row] * Bsub[col + w*4][k];\n" \
// "            }  \n" \
// "          }  \n" \
// "     \n" \
// "          barrier(CLK_LOCAL_MEM_FENCE);  \n" \
// "      }  \n" \
// "      for (int w=0; w<8; w++) {\n" \
// "        if((globalRow < M) && ((globalCol + w*4) < N)) {  \n" \
// "            C[(globalCol + w*4)*M + globalRow] = acc[w];  \n" \
// "        }\n" \
// "      } \n" \
// "}\n" \  
// "\n";

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

void transpose_scalar_block(const float *A, float *B, const int n, const int m, const int block_size_row, const int block_size_col) {
    for(int i=0; i<block_size_row; i++) {
        for(int j=0; j<block_size_col; j++) {
            B[j*n + i] = A[i*m +j];
        }
    }
}

void transpose_block(const float *A, float *B, const int n, const int m, const int block_size) {
    for(int i=0; i<n; i+=block_size) {
        for(int j=0; j<m; j+=block_size) {
            transpose_scalar_block(&A[i*m +j], &B[j*n + i], n, m, fmin(block_size,n-i), fmin(block_size,m-j));
        }
    }
}

void OpenCLPortableMatrixBatchVectorMultiplyAccumulate(const float* matrix,
                                                 int m_rows, int m_cols,
                                                 const float* vector,
                                                 int n_batch, float* result,
                                                 int result_stride,
                                                 cl_context context, cl_command_queue queue, cl_program program) {

  // Start Timers
  double wall0 = get_wall_time();
  double cpu0  = get_cpu_time();

  float* matrixT = (float*)malloc(m_rows*m_cols*sizeof(float));
  transpose_block(matrix, matrixT, m_rows, m_cols, 16);
  
  double wall1 = get_wall_time();
  double cpu1  = get_cpu_time();

  double wall = wall1 - wall0;
  double cpu = cpu1 - cpu0;
  
  __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "transpose: %lf", wall);

  cl_mem d_a;
  cl_mem d_b;
  cl_mem d_c;

  // cl_platform_id cpPlatform;
  // cl_device_id device_id;    
  // cl_context context;       
  // cl_command_queue queue;   
  // cl_program program;       
  cl_kernel kernel;

  size_t localSizetmp;
  cl_int err;

  // err = clGetPlatformIDs(1, &cpPlatform, NULL);
 
  // err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

  // clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE,  
  //        sizeof(localSizetmp), &localSizetmp, NULL);

  // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "localsize max: %d", localSizetmp);

  // globalSize = ceil(m_rows*m_cols/(localSizetmp*1.0))*localSize;

  // context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

  // queue = clCreateCommandQueue(context, device_id, 0, &err);

  // Start Timers
  // double wall0 = get_wall_time();
  // double cpu0  = get_cpu_time();

  // program = clCreateProgramWithSource(context, 1,
  //                         (const char **) & kernelSource, NULL, &err);

  // clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

  // double wall1 = get_wall_time();
  // double cpu1  = get_cpu_time();

  // double wall = wall1 - wall0;
  // double cpu = cpu1 - cpu0;
  
  // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "createprogram: %lf", wall);

  // wall0 = get_wall_time();
  // cpu0  = get_cpu_time();

  // Start Timers
  wall0 = get_wall_time();
  cpu0  = get_cpu_time();

  kernel = clCreateKernel(program, "matrixVectorMul", &err);

  d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, m_rows*m_cols*sizeof(float), NULL, NULL);
  d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, m_cols*n_batch*sizeof(float), NULL, NULL);
  d_c = clCreateBuffer(context, CL_MEM_READ_WRITE, m_rows*n_batch*sizeof(float), NULL, NULL);

  wall1 = get_wall_time();
  cpu1  = get_cpu_time();

  wall = wall1 - wall0;
  cpu = cpu1 - cpu0;
  
  __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "createkernelbuffer: %lf", wall);


  wall0 = get_wall_time();
  cpu0  = get_cpu_time();

  err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                                 m_rows*m_cols*sizeof(float), matrixT, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                                 m_cols*n_batch*sizeof(float), vector, 0, NULL, NULL);

  clFinish(queue);

  wall1 = get_wall_time();
  cpu1  = get_cpu_time();

  wall = wall1 - wall0;
  cpu = cpu1 - cpu0;
  
  __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "writebuffer: %lf", wall);

  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_c);
  err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_a);
  err  = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_b);
  err  = clSetKernelArg(kernel, 3, sizeof(int), &m_cols);
  err  = clSetKernelArg(kernel, 4, sizeof(int), &m_rows);
  err  = clSetKernelArg(kernel, 5, sizeof(int), &n_batch);

  const int TS = 32;
  // const size_t localSize = (size_t)TS;
  // const size_t globalSize0 = (size_t)(((m_rows-1)/TS+1)*TS);
  // const size_t globalSize1 = (size_t)(((n_batch-1)/TS+1)*TS);

  // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "GlobalSize0: %d", globalSize0);
  // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "GlobalSize1: %d", globalSize1);
  // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "localSize: %d", localSize);

  // const size_t local[2] = { localSize, localSize };
  // const size_t global[2] = { (size_t)(((m_rows-1)/TS+1)*TS), (size_t)(((n_batch-1)/TS+1)*TS) };

  // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "matrixsize: %d %d %d", m_rows, m_cols, n_batch);

  // const size_t local[2] = { (size_t) TS, (size_t) (TS/8) };
  // const size_t global[2] = { (size_t) (((m_rows-1)/32+1)*32), (size_t) (((n_batch-1)/32+1)*4) };

  const size_t local[2] = { 8, 32 };
  const size_t global[2] = { (size_t) (((m_rows-1)/8+1)*8), 32 };

  // Start Timers
  wall0 = get_wall_time();
  cpu0  = get_cpu_time();

  // err = clEnqueueTask(queue, kernel, 0, NULL,NULL);
  err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
  clFinish(queue);

  wall1 = get_wall_time();
  cpu1  = get_cpu_time();

  wall = wall1 - wall0;
  cpu = cpu1 - cpu0;
  
  __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "runkernel: %lf", wall);

  // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "Matmulerror: %d", err);

  // err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

  wall0 = get_wall_time();
  cpu0  = get_cpu_time();

  clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, m_rows*n_batch*sizeof(float), result, 0, NULL, NULL );

  clFinish(queue);

  wall1 = get_wall_time();
  cpu1  = get_cpu_time();

  wall = wall1 - wall0;
  cpu = cpu1 - cpu0;
  
  __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "readbuffer: %lf", wall);

  wall0 = get_wall_time();
  cpu0  = get_cpu_time();

  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);
  // clReleaseProgram(program);
  clReleaseKernel(kernel);
  // clReleaseCommandQueue(queue);
  // clReleaseContext(context);

  wall1 = get_wall_time();
  cpu1  = get_cpu_time();

  wall = wall1 - wall0;
  cpu = cpu1 - cpu0;
  
  __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "cleaning: %lf", wall);

}

void PortableMatrixBatchVectorMultiplyAccumulate(const float* matrix,
                                                 int m_rows, int m_cols,
                                                 const float* vector,
                                                 int n_batch, float* result,
                                                 int result_stride) {
  // vector per kolom
  // matrix per baris
  // result per kolom
  float* result_in_batch = result;
  for (int b = 0; b < n_batch; b++) {
    const float* matrix_ptr = matrix;
    for (int r = 0; r < m_rows; r++) {
      const float* vector_in_batch = vector + b * m_cols;
      for (int c = 0; c < m_cols; c++) {
        *result_in_batch += *matrix_ptr++ * *vector_in_batch++;
      }
      result_in_batch += result_stride;
    }
  }

  // //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "1");
}

namespace tflite {
namespace tensor_utils {

void NeonMatrixBatchVectorMultiplyAccumulate(const float* matrix, int m_rows,
                                             int m_cols, const float* vector,
                                             int n_batch, float* result,
                                             int result_stride) {
  // // Start Timers
  // double wall0 = get_wall_time();
  // double cpu0  = get_cpu_time();

  // PortableMatrixBatchVectorMultiplyAccumulate(matrix,m_rows,m_cols,vector,n_batch,result,1);
  // OpenCLPortableMatrixBatchVectorMultiplyAccumulate(matrix,m_rows,m_cols,vector,n_batch,result,1);
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "neon1");

  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start =
      m_cols - (m_cols & (kFloatWeightsPerNeonLane - 1));

  // The arrays used to cache the vector.
  float32x4_t* vector_cache_float32x4 =
      new float32x4_t[(m_cols / kFloatWeightsPerNeonLane) *
                      sizeof(float32x4_t)];
  const int kUnrollSize = 2;
  for (int b = 0; b < n_batch; b++) {
    float* result_in_batch = result + b * m_rows * result_stride;
    const float* vector_in_batch = vector + b * m_cols;

    const float* matrix_ptr0 = matrix;
    // If there is only 1 row, we don't want to assign an illegal pointer.
    const float* matrix_ptr1 = nullptr;
    if (m_rows > 1) {
      matrix_ptr1 = matrix + m_cols;
    }

    // Cahce the vector.
    for (int c = 0; c < postamble_start; c += kFloatWeightsPerNeonLane) {
      vector_cache_float32x4[c >> 2] = vld1q_f32(vector_in_batch + c);
    }

    // Main matrix by vector multiplication loop, which handles two rows of
    // matrix by vector multiplication.
    for (int r = 0; r < (m_rows & ~(kUnrollSize - 1)); r += kUnrollSize) {
      float32x4_t acc0_32x4 = vmovq_n_f32(0.0);
      float32x4_t acc1_32x4 = vmovq_n_f32(0.0);
      for (int c = 0; c < postamble_start; c += kFloatWeightsPerNeonLane) {
        float32x4_t temp = vector_cache_float32x4[c >> 2];
        // Load 4 float values from vector1 and vector2 and accumulator.
        float32x4_t v0_f32x4 = vld1q_f32(matrix_ptr0 + c);
        float32x4_t v1_f32x4 = vld1q_f32(matrix_ptr1 + c);
        // Vector multiply-accumulate 4 float
        acc0_32x4 = vmlaq_f32(acc0_32x4, v0_f32x4, temp);
        acc1_32x4 = vmlaq_f32(acc1_32x4, v1_f32x4, temp);
      }
      // Add the 4 intermediate sum values to get the final dot-prod value for
      // this column.
      *result_in_batch +=
          (vgetq_lane_f32(acc0_32x4, 0) + vgetq_lane_f32(acc0_32x4, 1) +
           vgetq_lane_f32(acc0_32x4, 2) + vgetq_lane_f32(acc0_32x4, 3));
      *(result_in_batch + result_stride) +=
          (vgetq_lane_f32(acc1_32x4, 0) + vgetq_lane_f32(acc1_32x4, 1) +
           vgetq_lane_f32(acc1_32x4, 2) + vgetq_lane_f32(acc1_32x4, 3));
      for (int c = postamble_start; c < m_cols; c++) {
        *result_in_batch += matrix_ptr0[c] * vector_in_batch[c];
        *(result_in_batch + result_stride) +=
            matrix_ptr1[c] * vector_in_batch[c];
      }
      matrix_ptr0 += kUnrollSize * m_cols;
      matrix_ptr1 += kUnrollSize * m_cols;
      result_in_batch += kUnrollSize * result_stride;
    }
    for (int r = (m_rows & ~(kUnrollSize - 1)); r < m_rows; r++) {
      float32x4_t acc0_32x4 = vmovq_n_f32(0.0);
      for (int c = 0; c < postamble_start; c += kFloatWeightsPerNeonLane) {
        float32x4_t temp = vector_cache_float32x4[c >> 2];
        // Load 4 float values from vector1 and vector2 and accumulator.
        float32x4_t v0_f32x4 = vld1q_f32(matrix_ptr0 + c);
        // Vector multiply-accumulate 4 float
        acc0_32x4 = vmlaq_f32(acc0_32x4, v0_f32x4, temp);
      }
      // Add the 4 intermediate sum values to get the final dot-prod value for
      // this column.
      *result_in_batch +=
          (vgetq_lane_f32(acc0_32x4, 0) + vgetq_lane_f32(acc0_32x4, 1) +
           vgetq_lane_f32(acc0_32x4, 2) + vgetq_lane_f32(acc0_32x4, 3));
      for (int c = postamble_start; c < m_cols; c++) {
        *result_in_batch += matrix_ptr0[c] * vector_in_batch[c];
      }
      matrix_ptr0 += m_cols;
      result_in_batch += result_stride;
    }
  }
  delete[] vector_cache_float32x4;

  // // Stop Timers
  // double wall1 = get_wall_time();
  // double cpu1  = get_cpu_time();

  // double wall = wall1 - wall0;
  // double cpu = cpu1 - cpu0;

  // float* h_a;
  // float* h_b;
  // float* h_c;

  // h_a = (float*)malloc(10000*sizeof(float));
  // h_b = (float*)malloc(10000*sizeof(float));
  // h_c = (float*)malloc(10000*sizeof(float));

  // for(int i = 0; i < 10000; i++ )
  //   {
  //       h_a[i] = 1.0;
  //       if(i<10000) {
  //         h_b[i] = 1.0;
  //       }
  //   }

  // // Start Timers
  // double wall0 = get_wall_time();
  // double cpu0  = get_cpu_time();

  // // PortableMatrixBatchVectorMultiplyAccumulate(h_a,100,100,h_b,100,h_c,1);
  // OpenCLPortableMatrixBatchVectorMultiplyAccumulate(h_a,100,100,h_b,100,h_c,1);

  // // Stop timers
  // double wall1 = get_wall_time();
  // double cpu1  = get_cpu_time();

  // double wall = wall1 - wall0;
  // double cpu = cpu1 - cpu0;

  // double sum = 0;
  // for(int i = 0; i < 10000; i++) {
  //   sum += h_c[i];
  // }

  // note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "Walltime: %lf", wall);
  // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "Sum: %lf", sum);
}

void NeonMatrixBatchVectorMultiplyAccumulateOpenCL(const float* matrix, int m_rows,
                                             int m_cols, const float* vector,
                                             int n_batch, float* result,
                                             int result_stride,
                                             cl_context context_cl, cl_command_queue queue, cl_program program) {
  // Start Timers
  double wall0 = get_wall_time();
  double cpu0  = get_cpu_time();

  // PortableMatrixBatchVectorMultiplyAccumulate(matrix,m_rows,m_cols,vector,n_batch,result,1);
  OpenCLPortableMatrixBatchVectorMultiplyAccumulate(matrix,m_rows,m_cols,vector,n_batch,result,1, context_cl, queue, program);

  // Stop timers
  double wall1 = get_wall_time();
  double cpu1  = get_cpu_time();

  double wall = wall1 - wall0;
  double cpu = cpu1 - cpu0;

  // double sum = 0;
  // for(int i = 0; i < 10000; i++) {
  //   sum += h_c[i];
  // }

  // note: andoird log
  __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "Walltime: %lf", wall);
  // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "Sum: %lf", sum);
}

void NeonVectorVectorCwiseProduct(const float* vector1, const float* vector2,
                                  int v_size, float* result) {
  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start =
      v_size - (v_size & (kFloatWeightsPerNeonLane - 1));
  for (int v = 0; v < postamble_start; v += kFloatWeightsPerNeonLane) {
    // Load 4 float values from vector1 and vector2.
    float32x4_t v1_f32x4 = vld1q_f32(vector1 + v);
    float32x4_t v2_f32x4 = vld1q_f32(vector2 + v);
    // Vector multiply 4 float
    float32x4_t mul_32x4 = vmulq_f32(v1_f32x4, v2_f32x4);
    // Save to result array.
    vst1q_f32(&result[v], mul_32x4);
  }
  for (int v = postamble_start; v < v_size; v++) {
    result[v] = vector1[v] * vector2[v];
  }
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "neon1");
}

void NeonVectorVectorCwiseProductAccumulate(const float* vector1,
                                            const float* vector2, int v_size,
                                            float* result) {
  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start =
      v_size - (v_size & (kFloatWeightsPerNeonLane - 1));
  for (int v = 0; v < postamble_start; v += kFloatWeightsPerNeonLane) {
    // Load 4 float values from vector1 and vector2 and accumulator.
    float32x4_t v1_f32x4 = vld1q_f32(vector1 + v);
    float32x4_t v2_f32x4 = vld1q_f32(vector2 + v);
    float32x4_t acc_32x4 = vld1q_f32(result + v);
    // Vector multiply-accumulate 4 float
    acc_32x4 = vmlaq_f32(acc_32x4, v1_f32x4, v2_f32x4);
    // Save to result array.
    vst1q_f32(&result[v], acc_32x4);
  }
  for (int v = postamble_start; v < v_size; v++) {
    result[v] += vector1[v] * vector2[v];
  }
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "neon1");
}

void NeonVectorBatchVectorCwiseProductAccumulate(const float* vector,
                                                 int v_size,
                                                 const float* batch_vector,
                                                 int n_batch, float* result) {
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "neon1");

  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start =
      v_size - (v_size & (kFloatWeightsPerNeonLane - 1));

  // The arrays used to cache the vector.
  float32x4_t* vector_cache_float32x4 =
      new float32x4_t[(v_size / kFloatWeightsPerNeonLane) *
                      sizeof(float32x4_t)];
  for (int v = 0; v < postamble_start; v += kFloatWeightsPerNeonLane) {
    vector_cache_float32x4[v >> 2] = vld1q_f32(vector + v);
  }

  float* result_ptr = result;
  const float* batch_vector_ptr = batch_vector;
  for (int b = 0; b < n_batch; b++) {
    for (int v = 0; v < postamble_start; v += kFloatWeightsPerNeonLane) {
      // Load from memory to vectors.
      float32x4_t result_f32x4 = vld1q_f32(result_ptr + v);
      float32x4_t batch_vector_f32x4 = vld1q_f32(batch_vector_ptr + v);
      // Multiply-accumulate.
      result_f32x4 = vmlaq_f32(result_f32x4, batch_vector_f32x4,
                               vector_cache_float32x4[v >> 2]);
      // Store.
      vst1q_f32(result_ptr + v, result_f32x4);
    }
    // Postamble loop
    for (int v = postamble_start; v < v_size; v++) {
      result_ptr[v] += vector[v] * batch_vector_ptr[v];
    }
    // Update the pointers.
    result_ptr += v_size;
    batch_vector_ptr += v_size;
  }
  delete[] vector_cache_float32x4;
}

void NeonSub1Vector(const float* vector, int v_size, float* result) {
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "neon1");

  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start =
      v_size - (v_size & (kFloatWeightsPerNeonLane - 1));

  float32x4_t one_f32x4 = vmovq_n_f32(1.0);
  for (int v = 0; v < postamble_start; v += kFloatWeightsPerNeonLane) {
    // Load 4 float values from the current pointers of the input column and
    // subtract from 1.
    float32x4_t v_f32x4 = vld1q_f32(vector + v);
    float32x4_t result_f32x4 = vsubq_f32(one_f32x4, v_f32x4);
    // Save to output.
    vst1q_f32(result + v, result_f32x4);
  }
  for (int v = postamble_start; v < v_size; v++) {
    result[v] = 1.0f - vector[v];
  }
}

void NeonClipVector(const float* vector, int v_size, float abs_limit,
                    float* result) {
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "neon1");

  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start =
      v_size - (v_size & (kFloatWeightsPerNeonLane - 1));

  // Replicate abs_limit and -abs_limit in two vectors.
  const float32x4_t abs_limit_f32x4 = vmovq_n_f32(abs_limit);
  const float32x4_t neg_abs_limit_f32x4 = vmovq_n_f32(-abs_limit);

  for (int v = 0; v < postamble_start; v += kFloatWeightsPerNeonLane) {
    // Load from memory to vector.
    float32x4_t v_f32x4 = vld1q_f32(vector + v);
    // Clip between abs_limit and -abs_limit.
    float32x4_t result_f32x4 = vminq_f32(abs_limit_f32x4, v_f32x4);
    result_f32x4 = vmaxq_f32(neg_abs_limit_f32x4, result_f32x4);
    // Save to output.
    vst1q_f32(result + v, result_f32x4);
  }
  // Postamble loop.
  for (int v = postamble_start; v < v_size; v++) {
    result[v] = (abs_limit < vector[v]) ? abs_limit : vector[v];
    result[v] = (-abs_limit > result[v]) ? -abs_limit : result[v];
  }
}

float NeonVectorVectorDotProduct(const float* vector1, const float* vector2,
                                 int v_size) {
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "neondot");
  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start =
      v_size - (v_size & (kFloatWeightsPerNeonLane - 1));
  float32x4_t acc_32x4 = vmovq_n_f32(0.0);
  for (int v = 0; v < postamble_start; v += kFloatWeightsPerNeonLane) {
    // Load 4 float values from vector1 and vector2 and accumulator.
    float32x4_t v1_f32x4 = vld1q_f32(vector1 + v);
    float32x4_t v2_f32x4 = vld1q_f32(vector2 + v);
    // Vector multiply-accumulate 4 float
    acc_32x4 = vmlaq_f32(acc_32x4, v1_f32x4, v2_f32x4);
  }

  float result = (vgetq_lane_f32(acc_32x4, 0) + vgetq_lane_f32(acc_32x4, 1) +
                  vgetq_lane_f32(acc_32x4, 2) + vgetq_lane_f32(acc_32x4, 3));
  // Postamble loop.
  for (int v = postamble_start; v < v_size; v++) {
    result += vector1[v] * vector2[v];
  }
  return result;
}

void NeonBatchVectorBatchVectorDotProduct(const float* vector1,
                                          const float* vector2, int v_size,
                                          int n_batch, float* result,
                                          int result_stride) {
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "neon1");
  float* result_ptr = result;
  const float* vector1_ptr = vector1;
  const float* vector2_ptr = vector2;
  for (int b = 0; b < n_batch; b++) {
    *result_ptr = NeonVectorVectorDotProduct(vector1_ptr, vector2_ptr, v_size);
    vector1_ptr += v_size;
    vector2_ptr += v_size;
    result_ptr += result_stride;
  }
}

void NeonReductionSumVector(const float* input_vector, float* output_vector,
                            int output_size, int reduction_size) {
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "neon1");
  const float* input_vector_ptr = input_vector;
  for (int o = 0; o < output_size; o++) {
    // If reduction_size is not divisible by kWeightsPerNeonLane, we cannot use
    // the main vectorized loop, and we need to process sequentially.
    // postamble_start shows the start index where this should happen.
    const int postamble_start =
        reduction_size - (reduction_size & (kFloatWeightsPerNeonLane - 1));
    float32x4_t sum_f32x4 = vmovq_n_f32(0.0);
    for (int r = 0; r < postamble_start; r += kFloatWeightsPerNeonLane) {
      float32x4_t v1_f32x4 = vld1q_f32(input_vector_ptr + r);
      sum_f32x4 = vaddq_f32(sum_f32x4, v1_f32x4);
    }
    output_vector[o] +=
        (vgetq_lane_f32(sum_f32x4, 0) + vgetq_lane_f32(sum_f32x4, 1) +
         vgetq_lane_f32(sum_f32x4, 2) + vgetq_lane_f32(sum_f32x4, 3));
    input_vector_ptr += postamble_start;

    // Postamble loop.
    for (int r = postamble_start; r < reduction_size; r++) {
      output_vector[o] += *input_vector_ptr++;
    }
  }
}

void NeonVectorShiftLeft(float* vector, int v_size, float shift_value) {
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "neon1");

  // This variable keeps track of the next to the last index which is being
  // copied to make sure we are not out of the vector boundary.
  int last_index_copy = kFloatWeightsPerNeonLane;
  int current_index_copy = 0;
  while (last_index_copy < v_size) {
    float32x4_t v_f32x4 = vld1q_f32(vector + current_index_copy + 1);
    vst1q_f32(vector + current_index_copy, v_f32x4);
    current_index_copy += kFloatWeightsPerNeonLane;
    last_index_copy += kFloatWeightsPerNeonLane;
  }
  // Postamble loop.
  for (int i = current_index_copy; i < v_size - 1; i++) {
    vector[i] = vector[i + 1];
  }
  vector[v_size - 1] = shift_value;
}

}  // namespace tensor_utils
}  // namespace tflite

#endif  // USE_NEON
