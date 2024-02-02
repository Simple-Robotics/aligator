/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "./lqr-problem.hpp"
#include "./blk-matrix.hpp"

#include <proxsuite-nlp/linalg/bunchkaufman.hpp>
#include <Eigen/LU>

#include "boost/core/make_span.hpp"
#include "tracy/Tracy.hpp"

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
template <typename Scalar> struct StageFactor {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using RowMatrixXs = Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>;

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

  StageFactor(uint nx, uint nu, uint nc, uint nth)
      : Qhat(nx, nx), Rhat(nu, nu), Shat(nx, nu), qhat(nx), rhat(nu),
        AtV(nx, nx), BtV(nu, nx), Gxhat(nx, nth), Guhat(nu, nth),
        ff({nu, nc, nx, nx}, {1}), fb({nu, nc, nx, nx}, {nx}),
        fth({nu, nc, nx, nx}, {nth}), kktMat({nu, nc}, {nu, nc}),
        kktChol(kktMat.rows()), Efact(nx), Ptilde(nx, nx), Einv(nx, nx),
        EinvP(nx, nx), schurMat(nx, nx), schurChol(nx), vm(nx, nth) {
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
    fth.setZero();
    kktMat.setZero();

    yff_pre.setZero(nx);
    A_pre.setZero(nx, nx);
    Yth_pre.setZero(nx, nth);
    Ptilde.setZero();
    Einv.setZero();
    EinvP.setZero();
    schurMat.setZero();
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

  BlkMatrix<VectorXs, 4, 1> ff;          //< feedforward gains
  BlkMatrix<RowMatrixXs, 4, 1> fb;       //< feedback gains
  BlkMatrix<RowMatrixXs, 4, 1> fth;      //< parameter feedback gains
  BlkMatrix<MatrixXs, 2, 2> kktMat;      //< reduced KKT matrix buffer
  Eigen::BunchKaufman<MatrixXs> kktChol; //< reduced KKT LDLT solver
  Eigen::PartialPivLU<MatrixXs> Efact;   //< LU decomp. of E matrix
  VectorXs yff_pre;
  MatrixXs A_pre;
  MatrixXs Yth_pre;
  MatrixXs Ptilde;   //< product Et.inv P * E.inv
  MatrixXs Einv;     //< product P * E.inv
  MatrixXs EinvP;    //< product P * E.inv
  MatrixXs schurMat; //< Dual-space Schur matrix
  Eigen::BunchKaufman<MatrixXs>
      schurChol; //< Cholesky decomposition of Schur matrix
  value_t vm;    //< cost-to-go parameters
};

// Implementation of a proximal riccati kernel.
template <typename Scalar> struct ProximalRiccatiKernel {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using RowMatrixXs = Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>;
  using RowMatrixRef = Eigen::Ref<RowMatrixXs>;
  using ConstRowMatrixRef = Eigen::Ref<const RowMatrixXs>;
  using KnotType = LQRKnotTpl<Scalar>;
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

  inline static void terminalSolve(const KnotType &model, const Scalar mueq,
                                   StageFactorType &d) {
    ZoneScoped;
    value_t &vc = d.vm;
    // fill cost-to-go matrix
    VectorRef kff = d.ff.blockSegment(0);
    VectorRef zff = d.ff.blockSegment(1);
    RowMatrixRef K = d.fb.blockRow(0);
    RowMatrixRef Z = d.fb.blockRow(1);
    RowMatrixRef Kth = d.fth.blockRow(0);
    RowMatrixRef Zth = d.fth.blockRow(1);

    Eigen::Transpose<const MatrixXs> Ct = model.C.transpose();

    if (model.nu == 0) {
      Z = model.C / mueq;
      zff = model.d / mueq;
      Zth.setZero();
    } else {
      d.kktMat(0, 0) = model.R;
      d.kktMat(0, 1) = model.D.transpose();
      d.kktMat(1, 0) = model.D;
      d.kktMat(1, 1).diagonal().setConstant(-mueq);
      d.kktChol.compute(d.kktMat.matrix());

      kff = -model.r;
      zff = -model.d;
      K = -model.S.transpose();
      Z = -model.C;

      auto ffview = d.ff.template topBlkRows<2>();
      auto fbview = d.fb.template topBlkRows<2>();
      d.kktChol.solveInPlace(ffview.matrix());
      d.kktChol.solveInPlace(fbview.matrix());

      if (model.nth > 0) {
        Kth = -model.Gu;
        Zth.setZero();
        auto fthview = d.fth.template topBlkRows<2>();
        d.kktChol.solveInPlace(fthview.matrix());
      }
    }

    vc.Pmat.noalias() = model.Q + Ct * Z;
    vc.pvec.noalias() = model.q + Ct * zff;

    if (model.nu > 0) {
      vc.Pmat.noalias() += model.S * K;
      vc.pvec.noalias() += model.S * kff;
    }

    if (model.nth > 0) {
      vc.Vxt = model.Gx;
      vc.Vxt.noalias() += K.transpose() * model.Gu;
      vc.Vtt = model.Gth;
      vc.Vtt.noalias() += model.Gu.transpose() * Kth;
      vc.vt = model.gamma;
      vc.vt.noalias() += model.Gu.transpose() * kff;
    }
  }

  inline static bool backwardImpl(boost::span<const KnotType> stages,
                                  const Scalar mudyn, const Scalar mueq,
                                  boost::span<StageFactorType> datas);

  /// Solve initial stage
  inline static void
  computeInitial(VectorRef x0, VectorRef lbd0, const kkt0_t &kkt0,
                 const std::optional<ConstVectorRef> &theta_);

  inline static void stageKernelSolve(const KnotType &model, StageFactorType &d,
                                      value_t &vn, const Scalar mudyn,
                                      const Scalar mueq);

  /// Forward sweep.
  inline static bool
  forwardImpl(boost::span<const KnotType> stages,
              boost::span<const StageFactorType> datas,
              boost::span<VectorXs> xs, boost::span<VectorXs> us,
              boost::span<VectorXs> vs, boost::span<VectorXs> lbdas,
              const std::optional<ConstVectorRef> &theta_ = std::nullopt);
};

} // namespace gar
} // namespace aligator

#include "./riccati-impl.hxx"

namespace aligator {
namespace gar {
#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct StageFactor<context::Scalar>;
extern template struct ProximalRiccatiKernel<context::Scalar>;
#endif
} // namespace gar
} // namespace aligator
