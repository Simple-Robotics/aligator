/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, 2023-2025 INRIA
/// @author Wilson Jallet
#pragma once

#include "aligator/context.hpp"
#include "aligator/gar/lqr-problem.hpp"
#include "blk-matrix.hpp"

#include "aligator/core/bunchkaufman.hpp"
#include <Eigen/LU>
#include <Eigen/Cholesky>

#include "aligator/third-party/boost/core/make_span.hpp"

#include <optional>

namespace aligator {
namespace gar {

/// Create a boost::span object from a vector and two indices.
template <class T, class A>
inline boost::span<T> make_span_from_indices(std::vector<T, A> &vec, size_t i0,
                                             size_t i1) {
  return boost::make_span(vec.data() + i0, i1 - i0);
}

/// @copybrief make_span_from_indices
template <class T, class A>
inline boost::span<const T> make_span_from_indices(const std::vector<T, A> &vec,
                                                   size_t i0, size_t i1) {
  return boost::make_span(vec.data() + i0, i1 - i0);
}

/// Per-node struct for all computations in the factorization.
template <typename _Scalar> struct StageFactor {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS_WITH_ROW_TYPES(Scalar);

  struct value_t {
    MatrixXs Pmat; //< Riccati matrix
    VectorXs pvec; //< Riccati bias
    MatrixXs Vxx;  //< "cost-to-go" matrix
    VectorXs vx;   //< "cost-to-go" gradient
    MatrixXs Vxt;
    MatrixXs Vtt;
    VectorXs vt;

    value_t(uint nx, uint nth)
        : Pmat(nx, nx), pvec(nx), Vxx(nx, nx), vx(nx), Vxt(nx, nth),
          Vtt(nth, nth), vt(nth) {
      Vxt.setZero();
      Vtt.setZero();
      vt.setZero();
    }
  };

  StageFactor(uint nx, uint nu, uint nc, uint nx2, uint nth);

  MatrixXs Qhat;
  MatrixXs Rhat;
  MatrixXs Shat;
  VectorXs qhat;
  VectorXs rhat;
  RowMatrixXs AtV;
  RowMatrixXs BtV;

  // Parametric
  MatrixXs Gxhat;
  MatrixXs Guhat;

  BlkMatrix<VectorXs, 4, 1> ff;          //< feedforward gains
  BlkMatrix<RowMatrixXs, 4, 1> fb;       //< feedback gains
  BlkMatrix<RowMatrixXs, 4, 1> fth;      //< parameter feedback gains
  BlkMatrix<MatrixXs, 2, 2> kktMat;      //< reduced KKT matrix buffer
  Eigen::BunchKaufman<MatrixXs> kktChol; //< reduced KKT LDLT solver
  Eigen::PartialPivLU<MatrixXs> Efact;   //< LU decomp. of E matrix
  VectorXs yff_pre;
  MatrixXs A_pre;
  MatrixXs Yth_pre;
  MatrixXs Ptilde;                //< product Et.inv P * E.inv
  MatrixXs Einv;                  //< product P * E.inv
  MatrixXs EinvP;                 //< product P * E.inv
  MatrixXs schurMat;              //< Dual-space Schur matrix
  Eigen::LLT<MatrixXs> schurChol; //< Cholesky decomposition of Schur matrix
  value_t vm;                     //< cost-to-go parameters
};

/// @brief Kernel for use in Riccati-like algorithms for the proximal LQ
/// subproblem.
template <typename Scalar> struct ProximalRiccatiKernel {
  ALIGATOR_DYNAMIC_TYPEDEFS_WITH_ROW_TYPES(Scalar);
  using KnotType = LqrKnotTpl<Scalar>;
  using StageFactorType = StageFactor<Scalar>;
  using value_t = typename StageFactor<Scalar>::value_t;

  struct kkt0_t {
    BlkMatrix<MatrixXs, 2, 2> mat;
    BlkMatrix<VectorXs, 2, 1> ff;
    BlkMatrix<RowMatrixXs, 2, 1> fth;
    Eigen::BunchKaufman<MatrixXs> chol{mat.rows()};
    kkt0_t(uint nx, uint nc, uint nth)
        : mat({nx, nc}, {nx, nc}), ff(mat.rowDims()),
          fth(mat.rowDims(), {nth}) {}
  };

  static void terminalSolve(typename KnotType::const_view_t model,
                            const Scalar mueq, StageFactorType &d);

  static bool backwardImpl(boost::span<const KnotType> stages,
                           const Scalar mudyn, const Scalar mueq,
                           boost::span<StageFactorType> datas);

  /// Solve initial stage
  static void computeInitial(VectorRef x0, VectorRef lbd0, const kkt0_t &kkt0,
                             const std::optional<ConstVectorRef> &theta_);

  static void stageKernelSolve(typename KnotType::const_view_t model,
                               StageFactorType &d, value_t &vn,
                               const Scalar mudyn, const Scalar mueq);

  /// Forward sweep.
  static bool
  forwardImpl(boost::span<const KnotType> stages,
              boost::span<const StageFactorType> datas,
              boost::span<VectorXs> xs, boost::span<VectorXs> us,
              boost::span<VectorXs> vs, boost::span<VectorXs> lbdas,
              const std::optional<ConstVectorRef> &theta_ = std::nullopt);
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct StageFactor<context::Scalar>;
extern template struct ProximalRiccatiKernel<context::Scalar>;
#endif

} // namespace gar
} // namespace aligator
