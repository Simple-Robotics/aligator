/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once
#include "aligator/context.hpp"
#include "./lqr-knot.hpp"

#include <Eigen/Cholesky>

namespace aligator {
namespace gar {

/// A sequential, regularized Riccati algorithm
// for proximal-regularized, constrained LQ problems.
template <typename Scalar> class ProximalRiccatiSolverBackward {
public:
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using RowMatrixXs = Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>;
  using knot_t = LQRKnot<Scalar>;
  using llt_ref_t = Eigen::LLT<MatrixRef>;

  struct value_t {
    MatrixXs Pmat;
    VectorXs pvec;
    RowMatrixXs EPinv;
    MatrixXs Lbmat; //< Dual-space system matrix
    MatrixXs Vmat;
    VectorXs vvec;

    value_t(uint nx)
        : Pmat(nx, nx), pvec(nx), EPinv(nx, nx),
          //
          Lbmat(nx, nx),
          //
          Vmat(Pmat), vvec(pvec) {}
  };

  /// Per-node struct for all computations in the factorization.
  struct Data {
    Data(uint nx, uint nu, uint nc)
        : nrows(nx + nu + nc), ff(nrows), fb(nrows, nx),
          Mmat{{nrows, nrows},
               Mmat.topLeftCorner(nu, nu),
               Mmat.bottomLeftCorner(nc, nu),
               Mmat.bottomRightCorner(nc, nc).diagonal()},
          hmlt(nx, nu), vm(nx) {}

    //// Feedback gains

    uint nrows;
    VectorXs ff; //< feedforward gain
    MatrixXs fb; //< feedback gain
    struct {
      MatrixXs data;
      MatrixRef Rt;
      MatrixRef Dt;
      VectorRef dual;
    } Mmat;

    struct hamil_t {
      MatrixXs Qhat, Rhat, Shat;
      VectorXs qhat, rhat;

      RowMatrixXs AtV;
      RowMatrixXs BtV;
      hamil_t(uint nx, uint nu)
          : Qhat(nx, nx), Rhat(nu, nu), Shat(nx, nu), qhat(nx), rhat(nu),
            AtV(nx, nx), BtV(nu, nx) {}
    } hmlt;

    //// Cost-to-go

    value_t vm;
  };

  ProximalRiccatiSolver(const std::vector<knot_t> &knots, Scalar mu)
      : knots(knots), mu(mu) {
    for (uint t = 0; t < horizon() + 1; t++) {
      const knot_t &knot = knots[t];
      uint nx = knot.nx;

      // interior knots : rhs in [r, h], nrows = nu + nc
      datas.push_back(Data{nx, knot.nu, knot.nc});
    }
  }

  size_t horizon() const noexcept { return knots.size() - 1; }

  bool checkProblem() const noexcept { return knots.back().nu == 0; }

  void computeKktTerms(const knot_t &model, Data &d, const value_t &vnext) {
    auto &hmlt = d.hmlt;
    hmlt.AtV.noalias() = model.A.transpose() * vnext.Vmat;
    hmlt.BtV.noalias() = model.B.transpose() * vnext.Vmat;
    hmlt.Qhat.noalias() = hmlt.AtV * model.A;
    hmlt.Qhat += model.Q;

    hmlt.Rhat.noalias() = hmlt.BtV * model.B;
    hmlt.Rhat += model.R;

    hmlt.Shat.noalias() = hmlt.AtV * model.B;
    hmlt.Shat += model.S;

    hmlt.qhat.noalias() = hmlt.AtV * vnext.vvec;
    hmlt.qhat += model.q;
    hmlt.rhat.noalias() = hmlt.BtV * vnext.vvec;
    hmlt.rhat += model.r;

    d.ff.head(model.nu) = hmlt.rhat;
    d.ff.tail(model.nc) = model.d;

    d.fb.topRows(model.nu) = hmlt.Shat.transpose();
    d.fb.bottomRows(model.nc) = model.C;
  }

  bool run() {
    if (!checkProblem())
      return false;

    // terminal node
    {
      ALIGATOR_NOMALLOC_BEGIN
      Data &dN = datas.back();
      const knot_t &term = knots.back();
      // fill cost-to-go matrix
      dN.Pmat.noalias() = term.C.transpose() * term.C;
      dN.Pmat = term.Q + dN.Pmat / mu;

      dN.pvec.noalias() = term.C.transpose() * term.d;
      dN.pvec = term.q + dN.pvec / mu;
      ALIGATOR_NOMALLOC_END
    }

    int t = horizon();
    for (; t >= 0; --t) {
      ALIGATOR_NOMALLOC_BEGIN
      Data &d = datas[t + 1];
      const knot_t &model = knots[t];
      // compute decomposition in-place
      llt_ref_t Pchol(d.Pmat);
      d.EPinv = model.E.transpose();
      Pchol.solveInPlace(d.EPinv);
      d.EPinv.transposeInPlace();
      d.Lbmat.noalias() = d.EPinv * model.E.transpose();
      d.Lbmat.diagonal().array() += mu;

      // compute decomposition in-place
      llt_ref_t Lamchol(d.Lbmat);
      d.Vmat.setIdentity();
      Lamchol.solveInPlace(d.Vmat); // evaluate inverse of Lambda
      d.vvec.noalias() = d.EPinv * d.pvec;
      d.vvec = model.f - d.vvec;

      // fill in hamiltonian
      computeKktTerms(model, d);
      ALIGATOR_NOMALLOC_END

      d.Mmat.Rt = d.hmlt.Rhat;
      d.Mmat.Ct = model.C;
      d.Mmat.dual.setConstant(-mu);
      d.Mmat = d.Mmat.template selfAdjointView<Eigen::Lower>();
      Eigen::LDLT<MatrixXs> ldlt(d.Mmat);
    }
  }

  std::vector<knot_t> knots;
  std::vector<Data> datas;
  Scalar mu;
};

template <typename Scalar> class ProximalRiccatiSolverForward {
public:
};

} // namespace gar
} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./riccati.txx"
#endif
