/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once
#include "aligator/context.hpp"
#include "./lqr-knot.hpp"
#include "./BlkMatrix.hpp"

#include <Eigen/Cholesky>

namespace aligator {
namespace gar {

/// A sequential, regularized Riccati algorithm
// for proximal-regularized, constrained LQ problems.
template <typename Scalar> class ProximalRiccatiSolver {
public:
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using RowMatrixXs = Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>;
  using RowMatrixRef = Eigen::Ref<RowMatrixXs>;
  using knot_t = LQRKnotTpl<Scalar>;
  using vecvec_t = std::vector<VectorXs>;

  struct value_t {
    MatrixXs Pmat;               //< Riccati matrix
    VectorXs pvec;               //< Riccati bias
    MatrixXs Lbmat;              //< Dual-space Schur matrix
    MatrixXs Vmat;               //< "cost-to-go" matrix
    VectorXs vvec;               //< "cost-to-go" gradient
    Eigen::LLT<MatrixXs> Pchol;  //< Cholesky decomposition of Pmat
    Eigen::LLT<MatrixXs> Lbchol; //< Cholesky decomposition of Lbmat
    value_t(uint nx)
        : Pmat(nx, nx), pvec(nx), Lbmat(nx, nx), //
          Vmat(nx, nx), vvec(nx), Pchol(nx), Lbchol(nx) {}
  };

  struct hmlt_t {
    MatrixXs Qhat;
    MatrixXs Rhat;
    MatrixXs Shat;
    VectorXs qhat;
    VectorXs rhat;
    RowMatrixXs AtV;
    RowMatrixXs BtV;
    hmlt_t(uint nx, uint nu)
        : Qhat(nx, nx), Rhat(nu, nu), Shat(nx, nu), //
          qhat(nx), rhat(nu), AtV(nx, nx), BtV(nu, nx) {}
  };

  struct error_t {
    Scalar lbda;
    Scalar pm;
    Scalar pv;
    Scalar fferr;
    Scalar fberr;
  };

  /// Per-node struct for all computations in the factorization.
  struct stage_factor_t {
    stage_factor_t(uint nx, uint nu, uint nc)
        : ff({nu, nc, nx, nx}, {1}), fb({nu, nc, nx, nx}, {nx}),
          kktMat({nu, nc}), kktChol(kktMat.rows()), hmlt(nx, nu), vm(nx), //
          PinvEt(nx, nx), Pinvp(nx),                                      //
          tmpClp(nx, nx) {
      ff.setZero();
      fb.setZero();
      kktMat.setZero();
    }

    BlkMatrix<VectorXs, 4, 1> ff;     //< feedforward gains
    BlkMatrix<RowMatrixXs, 4, 1> fb;  //< feedback gains
    BlkMatrix<MatrixXs, 2, 2> kktMat; //< reduced KKT matrix buffer
    Eigen::LDLT<MatrixXs> kktChol;    //< reduced KKT LDLT solver
    hmlt_t hmlt;                      //< stage system data
    value_t vm;                       //< cost-to-go parameters
    MatrixXs PinvEt;                  //< tmp buffer for \f$P^{-1}E^\top\f$
    VectorXs Pinvp;                   //< tmp buffer for \f$P^{-1}p\f$
    error_t err;                      //< numerical errors
    MatrixXs tmpClp;                  //< tmp buffer for state/co-state params
  };

  explicit ProximalRiccatiSolver(const LQRProblemTpl<Scalar> &problem)
      : datas(), kkt0(problem.stages[0].nx, (uint)problem.nc0()),
        problem(problem) {
    initialize();
  }

  ProximalRiccatiSolver(LQRProblemTpl<Scalar> &&problem) = delete;

  static void computeKktTerms(const knot_t &model, stage_factor_t &d,
                              const value_t &vnext);

  /// Backward sweep.
  bool backward(Scalar mudyn, Scalar mueq);
  /// Forward sweep.
  bool forward(vecvec_t &xs, vecvec_t &us, vecvec_t &vs, vecvec_t &lbdas) const;

  std::vector<stage_factor_t> datas;
  struct kkt0_t {
    BlkMatrix<MatrixXs, 2, 2> mat;
    BlkMatrix<VectorXs, 2, 1> rhs{mat.rowDims()};
    Eigen::LDLT<MatrixXs> chol{mat.rows()};
    kkt0_t(uint nx, uint nc) : mat({nx, nc}) {}
  } kkt0;

protected:
  void initialize() {
    auto N = uint(problem.horizon());
    auto &knots = problem.stages;
    datas.reserve(N + 1);
    for (uint t = 0; t <= N; t++) {
      const knot_t &knot = knots[t];
      datas.emplace_back(knot.nx, knot.nu, knot.nc);
    }
  }

  const LQRProblemTpl<Scalar> &problem;
};

} // namespace gar
} // namespace aligator

#include "./riccati.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./riccati.txx"
#endif
