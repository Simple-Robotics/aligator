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

  struct value_t {
    MatrixXs Pmat;  //< Riccati matrix
    VectorXs pvec;  //< Riccati bias
    MatrixXs Lbmat; //< Dual-space system matrix
    MatrixXs Vmat;  //< "cost-to-go" matrix
    VectorXs vvec;
    Eigen::LLT<MatrixXs> chol;
    value_t(uint nx)
        : Pmat(nx, nx), pvec(nx), Lbmat(nx, nx), //
          Vmat(nx, nx), vvec(nx), chol(nx) {}
  };

  struct kkt_t {
    uint nu, nc;
    MatrixXs data;
    Eigen::LDLT<MatrixXs> chol;
    kkt_t(uint nu, uint nc)
        : nu(nu), nc(nc), data(nu + nc, nu + nc), chol(nu + nc) {
      data.setZero();
    }
    MatrixRef R() { return data.topLeftCorner(nu, nu); };
    MatrixRef D() { return data.bottomLeftCorner(nc, nu); };
    auto dual() { return data.bottomRightCorner(nc, nc).diagonal(); }
  };

  struct hmlt_t {
    MatrixXs Qhat, Rhat, Shat;
    VectorXs qhat, rhat;
    RowMatrixXs AtV;
    RowMatrixXs BtV;
    hmlt_t(uint nx, uint nu)
        : Qhat(nx, nx), Rhat(nu, nu), Shat(nx, nu), //
          qhat(nx), rhat(nu), AtV(nx, nx), BtV(nu, nx) {}
  };

  /// Per-node struct for all computations in the factorization.
  struct stage_solve_data_t {
    stage_solve_data_t(uint nx, uint nu, uint nc)
        : ff(nu + nc), fb(nu + nc, nx), Mmat(nu, nc), hmlt(nx, nu), vm(nx),
          PinvEt(nx, nx), wvec(nx) {}

    VectorXs ff;     //< feedforward gain
    MatrixXs fb;     //< feedback gain
    kkt_t Mmat;      //< KKT matrix buffer
    hmlt_t hmlt;     //< stage system data
    value_t vm;      //< cost-to-go parameters
    MatrixXs PinvEt; //< tmp buffer for \f$EP^{-1}\f$
    VectorXs wvec;   //< tmp buffer for \f$-P^{-1}p\f$
  };

  ProximalRiccatiSolverBackward(const std::vector<knot_t> &knots)
      : knots(knots) {
    assert(knots.size() > 0);
    auto N = size_t(horizon());
    datas.reserve(N + 1);
    for (size_t t = 0; t <= N; t++) {
      const knot_t &knot = knots[t];
      datas.emplace_back(knot.nx, knot.nu, knot.nc);
    }
  }

  inline long horizon() const noexcept { return long(knots.size()) - 1; }

  void computeKktTerms(const knot_t &model, stage_solve_data_t &d,
                       const value_t &vnext);

  bool run(Scalar mudyn, Scalar mueq);

  std::vector<knot_t> knots;
  std::vector<stage_solve_data_t> datas;
};

/// Forward sweep algorithm.
template <typename Scalar> class ProximalRiccatiSolverForward {
public:
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using bwd_algo_t = ProximalRiccatiSolverBackward<Scalar>;
  using vecvec_t = std::vector<VectorXs>;
  using knot_t = LQRKnot<Scalar>;

  static bool run(bwd_algo_t &bwd, vecvec_t &xs, vecvec_t &us, vecvec_t &vs,
                  vecvec_t &lbdas);
};

} // namespace gar
} // namespace aligator

#include "./riccati.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./riccati.txx"
#endif
