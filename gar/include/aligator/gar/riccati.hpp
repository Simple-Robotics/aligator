/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "./lqr-problem.hpp"
#include "./blk-matrix.hpp"

#include <proxsuite-nlp/linalg/bunchkaufman.hpp>
#include <Eigen/Cholesky>

#include <boost/core/make_span.hpp>

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
    MatrixXs Vxx;                   //< "cost-to-go" matrix
    VectorXs vx;                    //< "cost-to-go" gradient
    Eigen::LLT<MatrixXs> Pchol;     //< Cholesky decomposition of Pmat
    Eigen::LLT<MatrixXs> schurChol; //< Cholesky decomposition of Schur matrix
    MatrixXs Vxt;
    MatrixXs Vtt;
    VectorXs vt;

    value_t(uint nx, uint nth)
        : Pmat(nx, nx), pvec(nx), schurMat(nx, nx), //
          Vxx(nx, nx), vx(nx), Pchol(nx), schurChol(nx), Vxt(nx, nth),
          Vtt(nth, nth), vt(nth) {
      Vxt.setZero();
      Vtt.setZero();
      vt.setZero();
    }
  };

  /// Per-node struct for all computations in the factorization.
  struct stage_factor_t {
    stage_factor_t(uint nx, uint nu, uint nc, uint nth)
        : Qhat(nx, nx), Rhat(nu, nu), Shat(nx, nu), qhat(nx), rhat(nu),
          AtV(nx, nx), BtV(nu, nx), Gxhat(nx, nth), Guhat(nu, nth),
          ff({nu, nc, nx, nx}, {1}), fb({nu, nc, nx, nx}, {nx}),
          fth({nu, nc, nx, nx}, {nth}),                       //
          kktMat({nu, nc}, {nu, nc}), kktChol(kktMat.rows()), //
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
  bool backward(const Scalar mudyn, const Scalar mueq);

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

  static void solveOneStage(const knot_t &model, stage_factor_t &d, value_t &vn,
                            const Scalar mudyn, const Scalar mueq) {

    // compute matrix expressions for the inverse
    computeMatrixTerms(model, mudyn, mueq, vn, d);

    VectorRef kff = d.ff.blockSegment(0);
    VectorRef zff = d.ff.blockSegment(1);
    VectorRef xi = d.ff.blockSegment(2);
    VectorRef a = d.ff.blockSegment(3);
    a = vn.Pchol.solve(-vn.pvec);

    vn.vx.noalias() = model.f + model.E * a;

    // fill feedback system
    d.qhat.noalias() = model.q + d.AtV * vn.vx;
    d.rhat.noalias() = model.r + d.BtV * vn.vx;
    kff = -d.rhat;
    zff = -model.d;

    RowMatrixRef K = d.fb.blockRow(0);
    RowMatrixRef Z = d.fb.blockRow(1);
    RowMatrixRef Xi = d.fb.blockRow(2);
    RowMatrixRef A = d.fb.blockRow(3);
    K = -d.Shat.transpose();
    Z = -model.C;
    BlkMatrix<VectorRef, 2, 1> ffview = d.ff.template topBlkRows<2>();
    BlkMatrix<RowMatrixRef, 2, 1> fbview = d.fb.template topBlkRows<2>();
    d.kktChol.solveInPlace(ffview.matrix());
    d.kktChol.solveInPlace(fbview.matrix());

    // set closed loop dynamics
    xi.noalias() = vn.Vxx * vn.vx;
    xi.noalias() += d.BtV.transpose() * kff;

    Xi.noalias() = vn.Vxx * model.A;
    Xi.noalias() += d.BtV.transpose() * K;

    a.noalias() -= d.PinvEt * xi;
    A.noalias() = d.PinvEt * Xi;
    A *= -1;

    value_t &vc = d.vm;
    Eigen::Transpose<const MatrixXs> Ct = model.C.transpose();
    vc.Pmat.noalias() = d.Qhat + d.Shat * K + Ct * Z;
    vc.pvec.noalias() = d.qhat + d.Shat * kff + Ct * zff;

    if (model.nth > 0) {
      RowMatrixRef Kth = d.fth.blockRow(0);
      RowMatrixRef Zth = d.fth.blockRow(1);
      RowMatrixRef Xith = d.fth.blockRow(2);
      RowMatrixRef Ath = d.fth.blockRow(3);

      // store -Pinv * L
      Ath = vn.Pchol.solve(-vn.Vxt);
      // store -V * E * Pinv * L
      Xith.noalias() = model.E * Ath;

      d.Gxhat.noalias() = model.Gx + d.AtV * Xith;
      d.Guhat.noalias() = model.Gu + d.BtV * Xith;

      // set rhs of 2x2 block system and solve
      Kth = -d.Guhat;
      Zth.setZero();
      BlkMatrix<RowMatrixRef, 2, 1> fthview = d.fth.template topBlkRows<2>();
      d.kktChol.solveInPlace(fthview.matrix());

      // substitute into Xith, Ath gains
      Xith.noalias() += model.B * Kth;
      vn.schurChol.solveInPlace(Xith);
      Ath.noalias() -= d.PinvEt * Xith;

      // update vt, Vxt, Vtt
      vc.vt = vn.vt + model.gamma;
      // vc.vt.noalias() += d.Guhat.transpose() * kff;
      vc.vt.noalias() += model.Gu.transpose() * kff;
      vc.vt.noalias() += vn.Vxt.transpose() * a;

      // vc.Vxt.noalias() = d.Gxhat + K.transpose() * d.Guhat;
      vc.Vxt = model.Gx;
      vc.Vxt.noalias() += K.transpose() * model.Gu;
      vc.Vxt.noalias() += A.transpose() * vn.Vxt;

      vc.Vtt = model.Gth + vn.Vtt;
      vc.Vtt.noalias() += model.Gu.transpose() * Kth;
      vc.Vtt.noalias() += vn.Vxt.transpose() * Ath;
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
        : mat({nx, nc}, {nx, nc}), ff(mat.rowDims()),
          fth(mat.rowDims(), {nth}) {}
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
