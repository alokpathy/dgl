/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/spmm.cu
 * \brief SPMM C APIs and definitions.
 */
#include <cutlass/gemm/device/gemm.h>
#include <dgl/array.h>
#include "./spmm.cuh"
#include "./ge_spmm.cuh"
#include "./functor.cuh"
#include "../../runtime/cuda/cuda_common.h"

// #define TIMING

namespace dgl {

using namespace cuda;

namespace aten {
namespace {

/*! \brief Call cuBLAS geam API for transpose operation for float and double. */
template <typename DType>
cublasStatus_t Xgeam(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,
    const DType* alpha, const DType* A, int lda,
    const DType* beta, const DType* B, int ldb,
    DType* C, int ldc) {
  LOG(INFO) << "Not supported dtype";
  return CUBLAS_STATUS_EXECUTION_FAILED;
}

template <>
cublasStatus_t Xgeam<float>(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,
    const float* alpha, const float* A, int lda,
    const float* beta, const float* B, int ldb,
    float* C, int ldc) {
  return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda,
      beta, B, ldb, C, ldc);
}

template <>
cublasStatus_t Xgeam<double>(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,
    const double* alpha, const double* A, int lda,
    const double* beta, const double* B, int ldb,
    double* C, int ldc) {
  return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda,
      beta, B, ldb, C, ldc);
}

/* \brief IndexSelect operator kernel implementation.
 * \note duplicate of IndexSelectKernel defined in array_index_select.cu
 */
template <typename DType, typename IdType>
__global__ void _IndexSelectKernel(
    const DType* __restrict__ in,
    const IdType* __restrict__ idx,
    DType* __restrict__ out,
    int n, int m) {
  int i = blockIdx.x;
  for (int j = threadIdx.x; j < m; j += blockDim.x)
    out[i * m + j] = in[idx[i] * m + j];
}

/* \brief Transpose operator kernel implementation.
 * \note not efficient but it's not a bottleneck, used for float16 dtype.
 */
template <typename DType>
__global__ void _TransposeKernel(
    const DType* __restrict__ in,
    DType* __restrict__ out,
    int n, int m) {
  int i = blockIdx.x;
  for (int j = threadIdx.x; j < m; j += blockDim.x)
    out[i * m + j] = in[j * n + i];
}

/*
 * \brief Tranpose the input matrix.
 * \param row number of rows of input matrix.
 * \param col number of columns of input matrix.
 */
template <typename DType>
void _Transpose(const DType* in, DType* out,
                int row, int col) {
  DType alpha = 1., beta = 0.;
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  if (!thr_entry->cublas_handle)
    CUBLAS_CALL(cublasCreate(&(thr_entry->cublas_handle)));
  CUBLAS_CALL(cublasSetStream(thr_entry->cublas_handle, thr_entry->stream));
  CUBLAS_CALL(Xgeam<DType>(
      thr_entry->cublas_handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      row, col,
      &alpha, in, col,
      &beta, nullptr, row,
      out, row));
}

/*
 * \brief Tranpose the input matrix for data type half.
 * \note cuBLAS has no geam API for half data type, fallback to our kernel.
 */
template <>
void _Transpose<half>(const half* in, half* out,
                      int row, int col) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int nt = FindNumThreads(row);
  int nb = col;
  CUDA_KERNEL_CALL(_TransposeKernel, nb, nt, 0, thr_entry->stream, in, out, col, row);
}

/*
 * \brief
 */
template <typename DType, typename IdType>
__global__ void _IndexSelectKernel(const DType* array, const IdType* index,
                                   int64_t length, DType* out) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    out[tx] = array[index[tx]];
    tx += stride_x;
  }
}

/* \brief IndexSelect operator.
 * \note duplicate of IndexSelect defined in array_op.h but it can
 *    not be applied to float16 dtype.
 */
template<typename DType, typename IdType>
NDArray _IndexSelect(NDArray array, NDArray index) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const DType* array_data = static_cast<DType*>(array->data);
  const IdType* idx_data = static_cast<IdType*>(index->data);
  const int64_t arr_len = array->shape[0];
  const int64_t len = index->shape[0];
  NDArray ret = NDArray::Empty({len}, array->dtype, array->ctx);
  if (len == 0)
    return ret;
  DType* ret_data = static_cast<DType*>(ret->data);
  const int nt = FindNumThreads(len);
  const int nb = (len + nt - 1) / nt;
  CUDA_KERNEL_CALL(_IndexSelectKernel, nb, nt, 0, thr_entry->stream,
      array_data, idx_data, len, ret_data);
  return ret;
}

}  // namespace

namespace cusparse {

#if CUDART_VERSION < 11000
template <typename DType>
cusparseStatus_t Xcsrmm2(cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const DType* alpha, const cusparseMatDescr_t descrA,
    const DType* csrValA, const int* csrRowPtrA, const int* csrColIndA,
    const DType* B, int ldb, const DType* beta, DType* C, int ldc) {
  LOG(INFO) << "Not supported dtype";
  return CUSPARSE_STATUS_EXECUTION_FAILED;
}

template <>
cusparseStatus_t Xcsrmm2<float>(cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const float* alpha, const cusparseMatDescr_t descrA,
    const float* csrValA, const int* csrRowPtrA, const int* csrColIndA,
    const float* B, int ldb, const float* beta, float* C, int ldc) {
  return cusparseScsrmm2(handle, transA, transB, m, n, k, nnz,
      alpha, descrA, csrValA, csrRowPtrA, csrColIndA,
      B, ldb, beta, C, ldc);
}

template <>
cusparseStatus_t Xcsrmm2<double>(cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const double* alpha, const cusparseMatDescr_t descrA,
    const double* csrValA, const int* csrRowPtrA, const int* csrColIndA,
    const double* B, int ldb, const double* beta, double* C, int ldc) {
  return cusparseDcsrmm2(handle, transA, transB, m, n, k, nnz,
      alpha, descrA, csrValA, csrRowPtrA, csrColIndA,
      B, ldb, beta, C, ldc);
}
#endif

/*! Cusparse implementation of SpMM on Csr format. */
template <typename DType, typename IdType>
void CusparseCsrmm2(
    const DLContext& ctx,
    const CSRMatrix& csr,
    const DType* B_data, const DType* A_data,
    DType* C_data,
    int x_length) {
  // We use csrmm2 to perform following operation:
  // C = A x B, where A is a sparse matrix in csr format, B is the dense matrix for node
  // feature tensor. However, since cusparse only supports column-major, while our tensor
  // is stored in row-major, the actual computation is:
  // C = trans(A x trans(B)).
  // Currently, we use cublasXgeam to implement transposition and allocate intermediate
  // workspace memory for this.
  const int m = csr.num_rows;
  const int n = x_length;
  const int k = csr.num_cols;
  const int nnz = csr.indices->shape[0];
  const DType alpha = 1.0;
  const DType beta = 0.0;
  // device
  auto device = runtime::DeviceAPI::Get(ctx);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, thr_entry->stream));
  // all one data array
  DType* valptr = nullptr;
  if (!A_data) {
    valptr = static_cast<DType*>(device->AllocWorkspace(ctx, nnz * sizeof(DType)));
    _Fill(valptr, nnz, static_cast<DType>(1.));
  }
#if CUDART_VERSION >= 11000
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  constexpr auto dtype = cuda_dtype<DType>::value;
  constexpr auto idtype = cusparse_idtype<IdType>::value;
  CUSPARSE_CALL(cusparseCreateCsr(&matA,
      m, k, nnz,
      static_cast<IdType*>(csr.indptr->data),
      static_cast<IdType*>(csr.indices->data),
      const_cast<DType*>(valptr? valptr : A_data),
      idtype, idtype,
      CUSPARSE_INDEX_BASE_ZERO, dtype));
  CUSPARSE_CALL(cusparseCreateDnMat(&matB,
      k, n, n,
      const_cast<DType*>(B_data), dtype, CUSPARSE_ORDER_ROW));
  CUSPARSE_CALL(cusparseCreateDnMat(&matC,
      m, n, n,
      C_data, dtype, CUSPARSE_ORDER_ROW));

  auto transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  size_t workspace_size;
  CUSPARSE_CALL(cusparseSpMM_bufferSize(
      thr_entry->cusparse_handle, transA, transB,
      &alpha, matA, matB, &beta, matC,
      dtype, CUSPARSE_SPMM_CSR_ALG2,
      &workspace_size));
  void* workspace = device->AllocWorkspace(ctx, workspace_size);
  CUSPARSE_CALL(cusparseSpMM(
      thr_entry->cusparse_handle, transA, transB,
      &alpha, matA, matB, &beta, matC,
      dtype, CUSPARSE_SPMM_CSR_ALG2,
      workspace));
  device->FreeWorkspace(ctx, workspace);

  CUSPARSE_CALL(cusparseDestroySpMat(matA));
  CUSPARSE_CALL(cusparseDestroyDnMat(matB));
  CUSPARSE_CALL(cusparseDestroyDnMat(matC));
#else
  // allocate matrix for temporary transposed output
  DType* trans_out = static_cast<DType*>(device->AllocWorkspace(ctx, m * n * sizeof(DType)));

  cusparseMatDescr_t descr;
  CUSPARSE_CALL(cusparseCreateMatDescr(&descr));
  CUSPARSE_CALL(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  CUSPARSE_CALL(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
  CUSPARSE_CALL(Xcsrmm2<DType>(
      thr_entry->cusparse_handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_TRANSPOSE,
      m, n, k, nnz, &alpha,
      descr, (valptr)? valptr : A_data,
      static_cast<int32_t*>(csr.indptr->data),
      static_cast<int32_t*>(csr.indices->data),
      B_data, n, &beta, trans_out, m));
  CUSPARSE_CALL(cusparseDestroyMatDescr(descr));
  // transpose the output matrix
  _Transpose(trans_out, C_data, n, m);
  device->FreeWorkspace(ctx, trans_out);
#endif
  if (valptr)
    device->FreeWorkspace(ctx, valptr);
}
}  // namespace cusparse

#define SWITCH_OP(op, Op, ...)                                      \
  do {                                                              \
    if ((op) == "add") {                                            \
      typedef cuda::binary::Add<DType> Op;                          \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "sub") {                                     \
      typedef cuda::binary::Sub<DType> Op;                          \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "mul") {                                     \
      typedef cuda::binary::Mul<DType> Op;                          \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "div") {                                     \
      typedef cuda::binary::Div<DType> Op;                          \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "copy_lhs") {                                \
      typedef cuda::binary::CopyLhs<DType> Op;                      \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "copy_rhs") {                                \
      typedef cuda::binary::CopyRhs<DType> Op;                      \
      { __VA_ARGS__ }                                               \
    } else {                                                        \
      LOG(FATAL) << "Unsupported SpMM binary operator: " << op;     \
    }                                                               \
  } while (0)

/*!
 * \brief Determine whether cusparse SpMM function is applicable.
 */
template <int bits, typename IdType>
inline bool cusparse_available() {
#if CUDART_VERSION < 11000
  if (std::is_same<IdType, int>::value)
    if (bits > 16)
      return true;
  return false;
#else
  if (bits == 16)
    return false;  // cusparse's SpMM on fp16 is slow, temporally disabled.
  return true;
#endif
}

__global__ void SetCsrOffsets(int *offsets, int M1, int N1, int M2, int N2) {
  int     id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  for (int i = id; i < M1; i += stride) {
    offsets[i] = i * N1;
  }

  for (int i = id + M1; i < M1 + M2 + 1; i += stride) {
    offsets[i] = M1 * N1 + (i - M1) * N2;
  }
}

__global__ void SetCsrColumns(int *columns, int M1, int N1, int M2, int N2) {
  int     id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  for (int i = id; i < M1 * N1; i += stride) {
    columns[i] = i % N1;
  }

  for (int i = id + M1 * N1; i < M1 * N1 + M2 * N2; i += stride) {
    columns[i] = N1 + (i - M1 * N1) % N2;
  }
}

// TODO: template this with generic DType
void fused_gemm(NDArray A1, NDArray B1, NDArray C1, int M1, int K1, int N1,
                    NDArray A2, NDArray B2, NDArray C2, int M2, int K2, int N2) {

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();

#ifdef TIMING
  cudaEvent_t total_start, total_stop;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventCreate(&total_start);
  cudaEventCreate(&total_stop);

  cudaEventRecord(total_start);
#endif
  cusparseSpMatDescr_t matA, matB, matC;
  void*  dBuffer1    = NULL, *dBuffer2   = NULL;
  size_t bufferSize1 = 0,    bufferSize2 = 0;

  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }

#ifdef TIMING
  cudaEventRecord(start);
#endif
  // Convert A into sparse matrix
  int *dA_csrOffsets, *dA_columns;
  float *dA_values;
  // CUDA_CALL( cudaMalloc(&dA_csrOffsets, (M + 1) * sizeof(int)) );
  // CUDA_CALL( cudaMalloc(&dA_columns, M * K * sizeof(int)) );
  int matA_numrows = M1 + M2;
  int matA_nnz = M1 * K1 + M2 * K2;
  CUDA_CALL( cudaMalloc(&dA_csrOffsets, (matA_numrows + 1) * sizeof(int)) );
  CUDA_CALL( cudaMalloc(&dA_columns, matA_nnz * sizeof(int)) );
  CUDA_CALL( cudaMalloc(&dA_values, matA_nnz * sizeof(float)) );

  const int nt_aoff = FindNumThreads(matA_numrows + 1);
  const int nb_aoff = (matA_numrows + 1 + nt_aoff - 1) / nt_aoff;

  const int nt_acol = FindNumThreads(matA_nnz);
  const int nb_acol = (matA_nnz + nt_acol - 1) / nt_acol;

  // CUDA_KERNEL_CALL( SetCsrOffsets, nb_aoff, nt_aoff, 0, thr_entry->stream, dA_csrOffsets, M, K );
  // CUDA_KERNEL_CALL( SetCsrColumns, nb_acol, nt_acol, 0, thr_entry->stream, dA_columns, M, K );
  // CUSPARSE_CALL( cusparseCreateCsr(&matA, M, K, M * K,
  //                                   dA_csrOffsets, dA_columns, A->data,
  //                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
  //                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );
  CUDA_KERNEL_CALL( SetCsrOffsets, nb_aoff, nt_aoff, 0, thr_entry->stream, dA_csrOffsets, M1, K1, M2, K2 );
  CUDA_KERNEL_CALL( SetCsrColumns, nb_acol, nt_acol, 0, thr_entry->stream, dA_columns, M1, K1, M2, K2 );
  CUDA_CALL( cudaMemcpy(dA_values, A1->data, M1 * K1 * sizeof(float), cudaMemcpyDeviceToDevice) );
  CUDA_CALL( cudaMemcpy(dA_values + M1 * K1, A2->data, M2 * K2 * sizeof(float), cudaMemcpyDeviceToDevice) );

  CUSPARSE_CALL( cusparseCreateCsr(&matA, M1 + M2, K1 + K2, matA_nnz,
                                    dA_csrOffsets, dA_columns, dA_values,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );

  // Convert B into sparse matrix
  int *dB_csrOffsets, *dB_columns;
  float *dB_values;
  // CUDA_CALL( cudaMalloc(&dB_csrOffsets, (K + 1) * sizeof(int)) );
  // CUDA_CALL( cudaMalloc(&dB_columns, K * N * sizeof(int)) );
  int matB_numrows = K1 + K2;
  int matB_nnz = K1 * N1 + K2 * N2;
  CUDA_CALL( cudaMalloc(&dB_csrOffsets, (matB_numrows + 1) * sizeof(int)) );
  CUDA_CALL( cudaMalloc(&dB_columns, matB_nnz * sizeof(int)) );
  CUDA_CALL( cudaMalloc(&dB_values, matB_nnz * sizeof(float)) );

  const int nt_boff = FindNumThreads(matB_numrows + 1);
  const int nb_boff = (matB_numrows + 1 + nt_boff - 1) / nt_boff;

  const int nt_bcol = FindNumThreads(matB_nnz);
  const int nb_bcol = (matB_nnz + nt_bcol - 1) / nt_bcol;

  // CUDA_KERNEL_CALL( SetCsrOffsets, nb_boff, nt_boff, 0, thr_entry->stream, dB_csrOffsets, K, N );
  // CUDA_KERNEL_CALL( SetCsrColumns, nb_bcol, nt_bcol, 0, thr_entry->stream, dB_columns, K, N );
  // CUSPARSE_CALL( cusparseCreateCsr(&matB, K, N, K * N,
  //                                   dB_csrOffsets, dB_columns, B->data,
  //                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
  //                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );
  CUDA_KERNEL_CALL( SetCsrOffsets, nb_boff, nt_boff, 0, thr_entry->stream, dB_csrOffsets, K1, N1, K2, N2 );
  CUDA_KERNEL_CALL( SetCsrColumns, nb_bcol, nt_bcol, 0, thr_entry->stream, dB_columns, K1, N1, K2, N2 );
  CUDA_CALL( cudaMemcpy(dB_values, B1->data, K1 * N1 * sizeof(float), cudaMemcpyDeviceToDevice) );
  CUDA_CALL( cudaMemcpy(dB_values + K1 * N1, B2->data, K2 * N2 * sizeof(float), cudaMemcpyDeviceToDevice) );
  CUSPARSE_CALL( cusparseCreateCsr(&matB, K1 + K2, N1 + N2, matB_nnz,
                                    dB_csrOffsets, dB_columns, dB_values,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );

  // Convert C into sparse matrix
  // CUSPARSE_CALL( cusparseCreateCsr(&matC, M, N, 0,
  //                                   NULL, NULL, NULL,
  //                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
  //                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );
  CUSPARSE_CALL( cusparseCreateCsr(&matC, M1 + M2, N1 + N2, 0,
                                    NULL, NULL, NULL,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );

  cusparseSpGEMMDescr_t spgemmDesc;
  CUSPARSE_CALL( cusparseSpGEMM_createDescr(&spgemmDesc) );

  // ask bufferSize1 bytes for external memory
  float               alpha       = 1.0f;
  float               beta        = 0.0f;
  cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
  CUSPARSE_CALL( cusparseSpGEMM_workEstimation(thr_entry->cusparse_handle, opA, opB,
                                        &alpha, matA, matB, &beta, matC,
                                        CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                        spgemmDesc, &bufferSize1, NULL) );
  CUDA_CALL( cudaMalloc((void**) &dBuffer1, bufferSize1) );

  // inspect the matrices A and B to understand the memory requirement for
  // the next step
  CUSPARSE_CALL( cusparseSpGEMM_workEstimation(thr_entry->cusparse_handle, opA, opB,
                                        &alpha, matA, matB, &beta, matC,
                                        CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                        spgemmDesc, &bufferSize1, dBuffer1) );

  // ask bufferSize2 bytes for external memory
  CUSPARSE_CALL( cusparseSpGEMM_compute(thr_entry->cusparse_handle, opA, opB,
                                 &alpha, matA, matB, &beta, matC,
                                 CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                 spgemmDesc, &bufferSize2, NULL) );
  CUDA_CALL( cudaMalloc((void**) &dBuffer2, bufferSize2) );

#ifdef TIMING
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float spgemm_preproc = 0;
  cudaEventElapsedTime(&spgemm_preproc, start, stop);

  cudaEventRecord(start);
#endif
  // compute the intermediate product of A * B
  CUSPARSE_CALL( cusparseSpGEMM_compute(thr_entry->cusparse_handle, opA, opB,
                                             &alpha, matA, matB, &beta, matC,
                                         CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                         spgemmDesc, &bufferSize2, dBuffer2) );
#ifdef TIMING
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float spgemm_compute = 0;
  cudaEventElapsedTime(&spgemm_compute, start, stop);

  cudaEventRecord(start);
#endif
  // allocate matrix C
  int *dC_csrOffsets, *dC_columns;
  float *dC_values;
  // CUDA_CALL( cudaMalloc((void**) &dC_csrOffsets, (M + 1) * sizeof(int)) );
  // CUDA_CALL( cudaMalloc((void**) &dC_columns, M * N * sizeof(int))   ); 
  int matC_numrows = M1 + M2;
  int matC_nnz = M1 * N1 + M2 * N2;
  CUDA_CALL( cudaMalloc((void**) &dC_csrOffsets, (matC_numrows + 1) * sizeof(int)) );
  CUDA_CALL( cudaMalloc((void**) &dC_columns, matC_nnz * sizeof(int))   ); 
  CUDA_CALL( cudaMalloc((void**) &dC_values, matC_nnz * sizeof(float))   ); 

  // update matC with the new pointers
  // CUSPARSE_CALL( cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, C->data) );
  CUSPARSE_CALL( cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values) );

  // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

  // copy the final products to the matrix C
  CUSPARSE_CALL( cusparseSpGEMM_copy(thr_entry->cusparse_handle, opA, opB,
                          &alpha, matA, matB, &beta, matC,
                          CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc) );

  // set C1 and C2's data to dC_values
  // Q: could we just set C1->data and C2->data pointers instead of a memcpy, or would that be a mem leak?
  CUDA_CALL( cudaMemcpy(C1->data, dC_values, M1 * N1 * sizeof(float), cudaMemcpyDeviceToDevice) );
  CUDA_CALL( cudaMemcpy(C2->data, dC_values + (M1 * N1), M2 * N2 * sizeof(float), cudaMemcpyDeviceToDevice) );

#ifdef TIMING
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float spgemm_copy = 0;
  cudaEventElapsedTime(&spgemm_copy, start, stop);

  cudaEventRecord(start);
#endif
  // destroy matrix/vector descriptors
  CUDA_CALL( cudaFree(dA_csrOffsets) );
  CUDA_CALL( cudaFree(dA_columns) );
  CUDA_CALL( cudaFree(dA_values) );
  CUDA_CALL( cudaFree(dB_csrOffsets) );
  CUDA_CALL( cudaFree(dB_columns) );
  CUDA_CALL( cudaFree(dB_values) );
  CUDA_CALL( cudaFree(dC_csrOffsets) );
  CUDA_CALL( cudaFree(dC_columns) );
  CUDA_CALL( cudaFree(dC_values) );
  CUDA_CALL( cudaFree(dBuffer1) );
  CUDA_CALL( cudaFree(dBuffer2) );

  CUSPARSE_CALL( cusparseSpGEMM_destroyDescr(spgemmDesc) );
  CUSPARSE_CALL( cusparseDestroySpMat(matA) );
  CUSPARSE_CALL( cusparseDestroySpMat(matB) );
  CUSPARSE_CALL( cusparseDestroySpMat(matC) );

#ifdef TIMING
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float spgemm_destroy = 0;
  cudaEventElapsedTime(&spgemm_destroy, start, stop);

  cudaEventRecord(total_stop);
  cudaEventSynchronize(total_stop);
  float spgemm_total = 0;
  cudaEventElapsedTime(&spgemm_total, total_start, total_stop);
  
  float spgemm_accum_time = spgemm_preproc + spgemm_compute + spgemm_copy + spgemm_destroy;
  printf("spgemm_total: %f %f\n", spgemm_total, spgemm_accum_time / spgemm_total);
  printf("spgemm_preproc: %f\n", spgemm_preproc);
  printf("spgemm_compute: %f\n", spgemm_compute);
  printf("spgemm_copy: %f\n", spgemm_copy);
  printf("spgemm_destroy: %f\n", spgemm_destroy); fflush(stdout);
#endif
}

/*!
 * \brief CUDA implementation of g-SpMM on Csr format.
 * \note use cusparse if the reduce operator is `sum` and there is
 *       no broadcast, use dgl's kernel in other cases.
 */
template <int XPU, typename IdType, int bits>
void SpMMCsr(const std::string& op, const std::string& reduce,
             const BcastOff& bcast,
             const CSRMatrix& csr,
             NDArray ufeat,
             NDArray efeat,
             NDArray out,
             std::vector<NDArray> out_aux) {
  int64_t feat_len = bcast.out_len;
  bool is_scalar_efeat = efeat.NumElements() == csr.indices->shape[0];
  bool use_efeat = op != "copy_lhs";

  if (reduce == "sum") {
    if (op == "copy_lhs" && cusparse_available<bits, IdType>()) {  // cusparse
      int64_t x_length = 1;
      for (int i = 1; i < ufeat->ndim; ++i)
        x_length *= ufeat->shape[i];
      SWITCH_BITS(bits, DType, {
        cusparse::CusparseCsrmm2<DType, IdType>(
            ufeat->ctx, csr,
            static_cast<DType*>(ufeat->data),
            nullptr,
            static_cast<DType*>(out->data),
            x_length);
      });
    } else if (op == "mul" && is_scalar_efeat && cusparse_available<bits, IdType>()) {  // cusparse
      int64_t x_length = 1;
      for (int i = 1; i < ufeat->ndim; ++i)
        x_length *= ufeat->shape[i];
      if (!IsNullArray(csr.data)) {
        SWITCH_BITS(bits, DType, {
          efeat = _IndexSelect<DType, IdType>(efeat, csr.data);
        });
      }
      SWITCH_BITS(bits, DType, {
        cusparse::CusparseCsrmm2<DType, IdType>(
            ufeat->ctx, csr,
            static_cast<DType*>(ufeat->data),
            static_cast<DType*>(efeat->data),
            static_cast<DType*>(out->data),
            x_length);
      });
    } else {  // general kernel
      SWITCH_BITS(bits, DType, {
        SWITCH_OP(op, Op, {
          cuda::SpMMCsr<IdType, DType, Op, cuda::reduce::Sum<IdType, DType> >(
              bcast, csr, ufeat, efeat, out, NullArray(), NullArray());
        });
      });
    }
  } else if (reduce == "max") {
    SWITCH_BITS(bits, DType, {
      SWITCH_OP(op, Op, {
        cuda::SpMMCsr<IdType, DType, Op, cuda::reduce::Max<IdType, DType> >(
            bcast, csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
      });
    });
  } else if (reduce == "min") {
    SWITCH_BITS(bits, DType, {
      SWITCH_OP(op, Op, {
        cuda::SpMMCsr<IdType, DType, Op, cuda::reduce::Min<IdType, DType> >(
            bcast, csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
      });
    });
  } else {
    LOG(FATAL) << "Not implemented";
  }
}


/*!
 * \brief CUDA implementation of g-SpMM on Coo format.
 */
template <int XPU, typename IdType, int bits>
void SpMMCoo(const std::string& op, const std::string& reduce,
             const BcastOff& bcast,
             const COOMatrix& coo,
             NDArray ufeat,
             NDArray efeat,
             NDArray out,
             std::vector<NDArray> out_aux) {
  
  if (reduce == "sum") {
    SWITCH_BITS(bits, DType, {
      SWITCH_OP(op, Op, {
        cuda::SpMMCoo<IdType, DType, Op, cuda::reduce::Sum<IdType, DType, true> > (
            bcast, coo, ufeat, efeat, out, NullArray(), NullArray());
      });
    });
  } else if (reduce == "max") {
    SWITCH_BITS(bits, DType, {
      SWITCH_OP(op, Op, {
        cuda::SpMMCoo<IdType, DType, Op, cuda::reduce::Max<IdType, DType, true> > (
            bcast, coo, ufeat, efeat, out, out_aux[0], out_aux[1]);
      });
    });
  }  else if (reduce == "min") {
    SWITCH_BITS(bits, DType, {
      SWITCH_OP(op, Op, {
        cuda::SpMMCoo<IdType, DType, Op, cuda::reduce::Min<IdType, DType, true> > (
            bcast, coo, ufeat, efeat, out, out_aux[0], out_aux[1]);
      });
    });
  } else {
    LOG(FATAL) << "Not implemented";
  }
}

template void SpMMCsr<kDLGPU, int32_t, 16>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCsr<kDLGPU, int64_t, 16>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCsr<kDLGPU, int32_t, 32>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCsr<kDLGPU, int64_t, 32>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCsr<kDLGPU, int32_t, 64>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCsr<kDLGPU, int64_t, 64>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);

template void SpMMCoo<kDLGPU, int32_t, 16>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLGPU, int64_t, 16>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLGPU, int32_t, 32>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLGPU, int64_t, 32>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLGPU, int32_t, 64>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLGPU, int64_t, 64>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);


}  // namespace aten
}  // namespace dgl
