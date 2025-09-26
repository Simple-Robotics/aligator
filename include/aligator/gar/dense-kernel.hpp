/// @file
/// @copyright Copyright (C) 2025 INRIA
#pragma once

#include "aligator/core/bunchkaufman.hpp"

#include "blk-matrix.hpp"
#include "lqr-problem.hpp"
#include <boost/core/span.hpp>

namespace aligator::gar {

/// @brief A dense Bunch-Kaufman based kernel.
template <typename _Scalar> struct DenseKernel {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS_WITH_ROW_TYPES(Scalar);
  using KnotType = LqrKnotTpl<Scalar>;

  struct Data {
    using BlkMat44 = BlkMatrix<MatrixXs, 4, 4>;
    using BlkRowMat41 = BlkMatrix<RowMatrixXs, 4, 1>;
    using BlkVec4 = BlkMatrix<VectorXs, 4, 1>;

    BlkMat44 kktMat;
    BlkRowMat41 fb;
    BlkRowMat41 ft;
    BlkVec4 ff;
    BunchKaufman<MatrixXs> ldl;

    Data(uint nx, uint nu, uint nc, uint nx2, uint nth)
        : kktMat({nu, nc, nx2, nx2}, {nu, nc, nx2, nx2})
        , fb({nu, nc, nx2, nx2}, {nx})
        , ft({nu, nc, nx2, nx2}, {nth})
        , ff({nu, nc, nx2, nx2})
        , ldl{nu + nc + 2 * nx2} {
      setZero();
    }

    void setZero() {
      kktMat.setZero();
      fb.setZero();
      ft.setZero();
      ff.setZero();
    }
  };

  // simple struct, can be copied
  struct value {
    MatrixRef Pxx;
    MatrixRef Pxt;
    MatrixRef Ptt;
    VectorRef px;
    VectorRef pt;
  };

  inline static void terminalSolve(const KnotType &knot, Data &d, value v,
                                   Scalar mueq) {
    d.kktMat.setZero();

    // assemble last-stage kkt matrix - includes input 'u'
    d.kktMat(0, 0) = knot.R;
    d.kktMat(0, 1) = knot.D.transpose();
    d.kktMat(1, 0) = knot.D;
    d.kktMat(1, 1).diagonal().array() = -mueq;

    VectorRef kff = d.ff[0] = -knot.r;
    VectorRef zff = d.ff[1] = -knot.d;
    RowMatrixRef K = d.fb.blockRow(0) = -knot.S.transpose();
    RowMatrixRef Z = d.fb.blockRow(1) = -knot.C;

    d.ldl.compute(d.kktMat.matrix());
    d.ldl.solveInPlace(d.ff.matrix());
    d.ldl.solveInPlace(d.fb.matrix());

    RowMatrixRef Kth = d.ft.blockRow(0) = -knot.Gu;
    RowMatrixRef Zth = d.ft.blockRow(1) = -knot.Gv;
    d.ldl.solveInPlace(d.ft.matrix());

    Eigen::Transpose Ct = knot.C.transpose();

    v.Pxx.noalias() = knot.Q + knot.S * K;
    v.Pxx.noalias() += Ct * Z;

    v.Pxt.noalias() = knot.Gx + K.transpose() * knot.Gu;
    v.Pxt.noalias() += Z.transpose() * knot.Gv;

    v.Ptt.noalias() = knot.Gth + knot.Gu.transpose() * Kth;
    v.Ptt.noalias() += knot.Gv.transpose() * Zth;

    v.px.noalias() = knot.q + knot.S * kff;
    v.px.noalias() += Ct * zff;

    v.pt.noalias() = knot.gamma + knot.Gu.transpose() * kff;
    v.pt.noalias() += knot.Gv.transpose() * zff;
  }

  inline static void stageKernelSolve(const KnotType &knot, Data &d, value v,
                                      const value *vn, Scalar mueq) {
    d.kktMat.setZero();
    d.kktMat(0, 0) = knot.R;
    d.kktMat(1, 0) = knot.D;
    d.kktMat(0, 1) = knot.D.transpose();
    d.kktMat(1, 1).diagonal().setConstant(-mueq);

    d.kktMat(2, 0) = knot.B;
    d.kktMat(0, 2) = knot.B.transpose();
    // d.kktMat(2, 2).setZero();
    d.kktMat(2, 3) = knot.E;
    d.kktMat(3, 2) = knot.E.transpose();
    if (vn)
      d.kktMat(3, 3) = vn->Pxx;

    // 1. factorize
    d.ldl.compute(d.kktMat.matrix());

    // 2. rhs
    // feedforward
    VectorRef kff = d.ff[0] = -knot.r;
    VectorRef zff = d.ff[1] = -knot.d;
    VectorRef lff = d.ff[2] = -knot.f;
    VectorRef yff = d.ff[3];
    if (vn)
      yff = -vn->px;

    // feedback
    RowMatrixRef K = d.fb.blockRow(0) = -knot.S.transpose();
    RowMatrixRef Z = d.fb.blockRow(1) = -knot.C;
    RowMatrixRef L = d.fb.blockRow(2) = -knot.A;
    RowMatrixRef Y = d.fb.blockRow(3).setZero();

    // parametric
    RowMatrixRef Kth = d.ft.blockRow(0) = -knot.Gu;
    RowMatrixRef Zth = d.ft.blockRow(1) = -knot.Gv;
    d.ft.blockRow(2).setZero();
    RowMatrixRef Yth = d.ft.blockRow(3);
    if (vn)
      Yth = -vn->Pxt;

    d.ldl.solveInPlace(d.ff.matrix());
    d.ldl.solveInPlace(d.fb.matrix());
    d.ldl.solveInPlace(d.ft.matrix());

    // 3. update value function
    Eigen::Transpose At = knot.A.transpose();
    Eigen::Transpose Ct = knot.C.transpose();

    v.Pxx.noalias() = knot.Q + knot.S * K;
    v.Pxx.noalias() += Ct * Z;
    v.Pxx.noalias() += At * L;

    v.Pxt = knot.Gx;
    v.Pxt.noalias() += K.transpose() * knot.Gu;
    v.Pxt.noalias() += Z.transpose() * knot.Gv;
    if (vn)
      v.Pxt.noalias() += Y.transpose() * vn->Pxt;

    v.Ptt = knot.Gth;
    v.Ptt.noalias() += Kth.transpose() * knot.Gu;
    v.Ptt.noalias() += Zth.transpose() * knot.Gv;
    if (vn)
      v.Ptt.noalias() += Yth.transpose() * vn->Pxt;

    v.px.noalias() = knot.q + knot.S * kff;
    v.px.noalias() += Ct * zff;
    v.px.noalias() += At * lff;

    v.pt.noalias() = knot.gamma + knot.Gu.transpose() * kff;
    v.pt.noalias() += knot.Gv.transpose() * zff;
    if (vn)
      v.pt.noalias() += vn->Pxt.transpose() * yff;
  }

  static bool forwardStep(size_t i, bool isTerminal, const KnotType &knot,
                          const Data &d, boost::span<VectorXs> xs,
                          boost::span<VectorXs> us, boost::span<VectorXs> vs,
                          boost::span<VectorXs> lbdas,
                          const std::optional<ConstVectorRef> &theta_) {
    ConstVectorRef kff = d.ff[0];
    ConstVectorRef zff = d.ff[1];
    ConstVectorRef lff = d.ff[2];
    ConstVectorRef yff = d.ff[3];

    ConstRowMatrixRef K = d.fb.blockRow(0);
    ConstRowMatrixRef Z = d.fb.blockRow(1);
    ConstRowMatrixRef L = d.fb.blockRow(2);
    ConstRowMatrixRef Y = d.fb.blockRow(3);

    ConstRowMatrixRef Kth = d.ft.blockRow(0);
    ConstRowMatrixRef Zth = d.ft.blockRow(1);
    ConstRowMatrixRef Lth = d.ft.blockRow(2);
    ConstRowMatrixRef Yth = d.ft.blockRow(3);

    if (knot.nu > 0)
      us[i].noalias() = kff + K * xs[i];
    vs[i].noalias() = zff + Z * xs[i];
    if (theta_.has_value()) {
      if (knot.nu > 0)
        us[i].noalias() += Kth * theta_.value();
      vs[i].noalias() += Zth * theta_.value();
    }

    if (isTerminal)
      return true;
    lbdas[i + 1].noalias() = lff + L * xs[i];
    xs[i + 1].noalias() = yff + Y * xs[i];
    if (theta_.has_value()) {
      lbdas[i + 1].noalias() += Lth * theta_.value();
      xs[i + 1].noalias() += Yth * theta_.value();
    }
    return true;
  }
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct DenseKernel<context::Scalar>;
#endif
} // namespace aligator::gar
