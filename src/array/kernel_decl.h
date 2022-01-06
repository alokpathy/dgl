/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/kernel_decl.h
 * \brief Sparse matrix format-specific operator declarations.
 */
#ifndef DGL_ARRAY_KERNEL_DECL_H_
#define DGL_ARRAY_KERNEL_DECL_H_

#include <dgl/bcast.h>
#include <dgl/base_heterograph.h>
#include <dgl/runtime/ndarray.h>

#include <string>
#include <vector>
#include <utility>

namespace dgl {
namespace aten {

// void fused_gemm(NDArray A, NDArray B, NDArray C, int M, int N, int K);
void fused_gemm(NDArray A1, NDArray B1, NDArray C1, int M1, int N1, int K1,
                    NDArray A2, NDArray B2, NDArray C2, int M2, int N2, int K2);

void fused_gemm_spmm(NDArray A, NDArray B, NDArray C, NDArray A_mats_rows, 
                        NDArray dA_csrOffsets, NDArray dA_columns, int M, int K, int N);

void fused_gemm_blockspmm(NDArray A, NDArray B, NDArray C, NDArray A_mats_rows, int M, int K, int N, 
                                int block_dim, int num_rels);

// void capi_gemms(std::vector<NDArray> A_mats, std::vector<NDArray> B_mats, std::vector<NDArray> C_mats, 
//                        std::vector<int> A_mats_rows, int middim, int outcol);

void capi_gemms(NDArray A_mats, NDArray B_mats, NDArray C_mats, 
                        NDArray A_mats_rows, int middim, int outcol, int num_rels, int total_edges);

void capi_gemms_half(NDArray A_mats, NDArray B_mats, NDArray C_mats, 
                        NDArray A_mats_rows, int middim, int outcol, int num_rels, int total_edges);

void pad_a(NDArray A3D, NDArray A_mats, NDArray A_mats_rows, int dim0, int dim1, int dim2);

void pad_a2d(NDArray A_pad, NDArray A_mats, NDArray A_mats_rows, NDArray dA_mats_rows, 
                NDArray A_pad_rows_ps, NDArray A_mat_rows_ps, 
                int dim0, int dim1, int num_edges, int num_rels);

void unpad_c(NDArray C3D, NDArray C_mats, NDArray C_mats_rows, int dim0, int dim1, int dim2);

void unpad_c2d(NDArray C_mats, NDArray C_pad, NDArray C_mats_rows, NDArray dC_mats_rows, NDArray C_pad_rows_ps,
                    NDArray C_mat_rows_ps, int dim0, int dim1, int block_dim, int num_rels);

int compute_pad(NDArray padding_arr, NDArray dA_mats_rows, int block_dim, int num_rels);

void pad_blockspmm(NDArray A_pad, NDArray A_mats, NDArray B_pad, NDArray C_pad, NDArray A_mats_rows, 
                    NDArray dA_mats_rows, NDArray padding_arr, 
                    int num_edges, int M, int K, int N, int num_rels, NDArray C_mats);

/*!
 * \brief Generalized Sparse Matrix Dense Matrix Multiplication on Csr format.
 */
template <int XPU, typename IdType, int bits>
void SpMMCsr(const std::string& op, const std::string& reduce,
             const BcastOff& bcast,
             const aten::CSRMatrix& csr,
             NDArray ufeat,
             NDArray efeat,
             NDArray out,
             std::vector<NDArray> out_aux);

/*!
 * \brief Generalized Sparse Matrix Dense Matrix Multiplication on Coo format.
 */
template <int XPU, typename IdType, int bits>
void SpMMCoo(const std::string& op, const std::string& reduce,
             const BcastOff& bcast,
             const aten::COOMatrix& coo,
             NDArray ufeat,
             NDArray efeat,
             NDArray out,
             std::vector<NDArray> out_aux);

/*!
 * \brief Generalized Sampled Dense-Dense Matrix Multiplication on Csr format.
 */
template <int XPU, typename IdType, int bits>
void SDDMMCsr(const std::string& op,
              const BcastOff& bcast,
              const aten::CSRMatrix& csr,
              NDArray lhs,
              NDArray rhs,
              NDArray out,
              int lhs_target,
              int rhs_target);

/*!
 * \brief Generalized Sampled Dense-Dense Matrix Multiplication on Coo format.
 */
template <int XPU, typename IdType, int bits>
void SDDMMCoo(const std::string& op,
              const BcastOff& bcast,
              const aten::COOMatrix& coo,
              NDArray lhs,
              NDArray rhs,
              NDArray out,
              int lhs_target,
              int rhs_target);

/*!
 * \brief Segment reduce.
 */
template <int XPU, typename IdType, int bits>
void SegmentReduce(const std::string& op,
                   NDArray feat,
                   NDArray offsets,
                   NDArray out,
                   NDArray arg);

/*!
 * \brief Scatter Add on first dimension.
 */
template <int XPU, typename IdType, int bits>
void ScatterAdd(NDArray feat,
                NDArray idx,
                NDArray out);

/*!
 * \brief Backward function of segment cmp.
 */
template <int XPU, typename IdType, int bits>
void BackwardSegmentCmp(NDArray feat,
                        NDArray arg,
                        NDArray out);

/*!
 * \brief Sparse-sparse matrix multiplication
 *
 * \param A The left operand.
 * \param A_weights The weights of matrix as a 1D tensor.
 * \param B The right operand.
 * \param B_weights The weights of matrix as a 1D tensor.
 *
 * \note GPU implementation will cast the indices to 32 bit.
 * \note The zero entries in the result are not removed.
 * \note The CSR matrix should not have duplicate entries.
 */
template <int XPU, typename IdType, typename DType>
std::pair<CSRMatrix, NDArray> CSRMM(
    const CSRMatrix& A,
    NDArray A_weights,
    const CSRMatrix& B,
    NDArray B_weights);

/*!
 * \brief Sparse-sparse matrix summation.
 *
 * \param A The sparse matrices with the same size.
 * \param A_weights The weights of each sparse matrix as a 1D tensor.
 *
 * \note GPU implementation will cast the indices to 32 bit.
 * \note The zero entries in the result are not removed.
 * \note The CSR matrix should not have duplicate entries.
 */
template <int XPU, typename IdType, typename DType>
std::pair<CSRMatrix, NDArray> CSRSum(
    const std::vector<CSRMatrix>& A,
    const std::vector<NDArray>& A_weights);

}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_KERNEL_DECL_H_
