/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "./lqr-problem.hpp"
#include "./blk-matrix.hpp"

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
  using ConstRowMatrixRef = Eigen::Ref<const RowMatrixXs>;
  using knot_t = LQRKnotTpl<Scalar>;

  struct value_t {
    MatrixXs Pmat;                  //< Riccati matrix
    VectorXs pvec;                  //< Riccati bias
    MatrixXs schurMat;              //< Dual-space Schur matrix
    MatrixXs Vmat;                  //< "cost-to-go" matrix
    VectorXs vvec;                  //< "cost-to-go" gradient
    Eigen::LLT<MatrixXs> Pchol;     //< Cholesky decomposition of Pmat
    Eigen::LLT<MatrixXs> schurChol; //< Cholesky decomposition of Schur matrix
    MatrixXs Lmat;
    MatrixXs Psi;
    VectorXs svec;

    value_t(uint nx, uint nth)
        : Pmat(nx, nx), pvec(nx), schurMat(nx, nx), //
          Vmat(nx, nx), vvec(nx), Pchol(nx), schurChol(nx), Lmat(nx, nth),
          Psi(nth, nth), svec(nth) {
      Lmat.setZero();
      Psi.setZero();
      svec.setZero();
    }
  };

  /// Per-node struct for all computations in the factorization.
  struct stage_factor_t {
    stage_factor_t(uint nx, uint nu, uint nc, uint nth)
        : Qhat(nx, nx), Rhat(nu, nu), Shat(nx, nu), qhat(nx), rhat(nu),
          AtV(nx, nx), BtV(nu, nx), Gxhat(nx, nth), Guhat(nu, nth),
          ff({nu, nc, nx, nx}, {1}), fb({nu, nc, nx, nx}, {nx}),
          fth({nu, nc, nx, nx}, {nth}),             //
          kktMat({nu, nc}), kktChol(kktMat.rows()), //
          vm(nx, nth), PinvEt(nx, nx) {
      Qhat.setZero();
      Rhat.setZero();
      Shat.setZero();
      qhat.setZero();
      rhat.setZero();

      AtV.setZero();
      BtV.setZero();

      Gxhat.setZero();
      Guhat.setZero();

      ff.setZero();
      fb.setZero();
      kktMat.setZero();
      fth.setZero();
    }

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

    BlkMatrix<VectorXs, 4, 1> ff;     //< feedforward gains
    BlkMatrix<RowMatrixXs, 4, 1> fb;  //< feedback gains
    BlkMatrix<RowMatrixXs, 4, 1> fth; //< parameter feedback gains
    BlkMatrix<MatrixXs, 2, 2> kktMat; //< reduced KKT matrix buffer
    Eigen::LDLT<MatrixXs> kktChol;    //< reduced KKT LDLT solver
    value_t vm;                       //< cost-to-go parameters
    MatrixXs PinvEt;                  //< tmp buffer for \f$P^{-1}E^\top\f$
  };

  explicit ProximalRiccatiSolver(const LQRProblemTpl<Scalar> &problem,
                                 bool solve_initial = true)
      : datas(), kkt0(problem.stages[0].nx, problem.nc0(), problem.ntheta()),
        thGrad(problem.ntheta()), thHess(problem.ntheta(), problem.ntheta()),
        solveInitial(solve_initial), problem(problem) {
    initialize();
  }

  ProximalRiccatiSolver(LQRProblemTpl<Scalar> &&problem) = delete;

  static void computeMatrixTerms(const knot_t &model, Scalar mudyn, Scalar mueq,
                                 value_t &vnext, stage_factor_t &d);

  /// Backward sweep.
  bool backward(Scalar mudyn, Scalar mueq);

  /// Solve initial stage
  void computeInitial(VectorRef x0, VectorRef lbd0,
                      const boost::optional<ConstVectorRef> &theta_) const {
    assert(kkt0.chol.info() == Eigen::Success);
    x0 = kkt0.ff.blockSegment(0);
    lbd0 = kkt0.ff.blockSegment(1);
    if (theta_.has_value()) {
      x0.noalias() += kkt0.fth.blockRow(0) * theta_.value();
      lbd0.noalias() += kkt0.fth.blockRow(1) * theta_.value();
    }
  }

  /// Forward sweep.
  bool
  forward(std::vector<VectorXs> &xs, std::vector<VectorXs> &us,
          std::vector<VectorXs> &vs, std::vector<VectorXs> &lbdas,
          const boost::optional<ConstVectorRef> &theta = boost::none) const;

  std::vector<stage_factor_t> datas;
  struct kkt0_t {
    BlkMatrix<MatrixXs, 2, 2> mat;
    BlkMatrix<VectorXs, 2, 1> ff;
    BlkMatrix<RowMatrixXs, 2, 1> fth;
    Eigen::LDLT<MatrixXs> chol{mat.rows()};
    kkt0_t(uint nx, uint nc, uint nth)
        : mat({nx, nc}), ff(mat.rowDims()), fth(mat.rowDims(), {nth}) {}
  } kkt0;

  VectorXs thGrad;   //< optimal value gradient wrt parameter
  MatrixXs thHess;   //< optimal value Hessian wrt parameter
  bool solveInitial; //< Whether to solve the initial stage

protected:
  void initialize();
  const LQRProblemTpl<Scalar> &problem;
};

} // namespace gar
} // namespace aligator

#include "./riccati.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./riccati.txx"
#endif
