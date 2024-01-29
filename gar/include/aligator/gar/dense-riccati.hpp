#pragma once

#include <proxsuite-nlp/linalg/bunchkaufman.hpp>

#include "riccati-base.hpp"

namespace aligator::gar {

/// A stagewise-dense Riccati solver
template <typename _Scalar>
class RiccatiSolverDense : public RiccatiSolverBase<_Scalar> {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = RiccatiSolverBase<Scalar>;
  using BlkMat44 = BlkMatrix<MatrixXs, 4, 4>;
  using BlkMat41 = BlkMatrix<MatrixXs, 4, 1>;
  using BlkVec4 = BlkMatrix<VectorXs, 4, 1>;
  using KnotType = LQRKnotTpl<Scalar>;

  struct FactorData {
    BlkMat44 kkt;
    BlkVec4 ff;
    BlkMat41 fb;
    Eigen::BunchKaufman<MatrixXs> ldl;
  };

  static FactorData init_factor(const LQRKnotTpl<Scalar> &knot) {
    std::array<long, 4> dims = {knot.nu, knot.nc, knot.nx2, knot.nx2};
    using ldl_t = decltype(FactorData::ldl);
    long ntot = std::accumulate(dims.begin(), dims.end(), 0);
    return FactorData{BlkMat44(dims, dims), BlkVec4(dims, {1}),
                      BlkMat41(dims, {knot.nx}), ldl_t{ntot}};
  }

  std::vector<FactorData> datas;
  std::vector<MatrixXs> Ps;
  std::vector<VectorXs> ps;
  struct {
    BlkMatrix<MatrixXs, 2, 2> mat;
    BlkMatrix<VectorXs, 2, 1> rhs;
    Eigen::BunchKaufman<MatrixXs> ldl;
  } kkt0;

  explicit RiccatiSolverDense(const LQRProblemTpl<Scalar> &problem)
      : Base(), problem_(&problem) {
    initialize();
  }

  void initialize() {
    auto N = (uint)problem_->horizon();
    const auto &stages = problem_->stages;
    Ps.resize(N + 1);
    ps.resize(N + 1);
    for (uint t = 0; t <= N; t++) {
      uint nx = stages[t].nx;
      Ps[t].setZero(nx, nx);
      ps[t].setZero(nx);
      datas.emplace_back(init_factor(stages[t]));
    }

    uint nx0 = stages[0].nx;
    std::array<long, 2> dims0 = {nx0, problem_->nc0()};
    kkt0 = {decltype(kkt0.mat)(dims0, dims0), decltype(kkt0.rhs)(dims0, {1}),
            Eigen::BunchKaufman<MatrixXs>(nx0 + problem_->nc0())};
  }

  bool backward(const Scalar mudyn, const Scalar mueq) {
    ZoneScoped;
    const std::vector<KnotType> &stages = problem_->stages;

    const uint N = (uint)problem_->horizon();
    {
      const KnotType &knot = stages[N];
      FactorData &fac = datas[N];
      fac.kkt.setZero();
      VectorRef kff = fac.ff[0];
      VectorRef zff = fac.ff[1];
      MatrixRef K = fac.fb.blockRow(0);
      MatrixRef Z = fac.fb.blockRow(1);

      // assemble last-stage kkt matrix - includes input 'u'
      fac.kkt(0, 0) = knot.R;
      fac.kkt(0, 1) = knot.D.transpose();
      fac.kkt(1, 0) = knot.D;
      fac.kkt(1, 1).diagonal().array() = -mueq;

      kff = -knot.r;
      zff = -knot.d;
      K = -knot.S.transpose();
      Z = -knot.C;

      fac.ldl.compute(fac.kkt.matrix());
      fac.ldl.solveInPlace(fac.ff.matrix());
      fac.ldl.solveInPlace(fac.fb.matrix());

      Eigen::Transpose Ct = knot.C.transpose();

      Ps[N].noalias() = knot.Q + knot.S * K;
      Ps[N].noalias() += Ct * Z;
      ps[N].noalias() = knot.q + knot.S * kff;
      ps[N].noalias() += Ct * zff;
    }

    uint i = N - 1;
    while (true) {
      const KnotType &knot = stages[i];
      FactorData &fac = datas[i];

      fac.kkt.setZero();
      fac.kkt(0, 0) = knot.R;
      fac.kkt(1, 0) = knot.D;
      fac.kkt(0, 1) = knot.D.transpose();
      fac.kkt(1, 1).diagonal().array() = -mueq;

      fac.kkt(2, 0) = knot.B;
      fac.kkt(0, 2) = knot.B.transpose();
      fac.kkt(2, 2).diagonal().array() = -mudyn;
      fac.kkt(3, 2) = knot.E.transpose();
      fac.kkt(2, 3) = knot.E;
      fac.kkt(3, 3) = Ps[i + 1];

      fac.ff[0] = -knot.r;
      fac.ff[1] = -knot.d;
      fac.ff[2] = -knot.f;
      fac.ff[3] = -ps[i + 1];

      fac.fb.blockRow(0) = -knot.S.transpose();
      fac.fb.blockRow(1) = -knot.C;
      fac.fb.blockRow(2) = -knot.A;
      fac.fb.blockRow(3).setZero();

      fac.ldl.compute(fac.kkt.matrix());
      fac.ldl.solveInPlace(fac.ff.matrix());
      fac.ldl.solveInPlace(fac.fb.matrix());

      Eigen::Transpose At = knot.A.transpose();
      Eigen::Transpose Ct = knot.C.transpose();
      Ps[i].noalias() = knot.Q + knot.S * fac.fb.blockRow(0);
      Ps[i].noalias() += Ct * fac.fb.blockRow(1);
      Ps[i].noalias() += At * fac.fb.blockRow(2);
      ps[i].noalias() = knot.q + knot.S * fac.ff[0];
      ps[i].noalias() += Ct * fac.ff[1];
      ps[i].noalias() += At * fac.ff[2];

      if (i == 0)
        break;
      i--;
    }

    // initial stage
    kkt0.mat(0, 0) = Ps[0];
    kkt0.mat(0, 1) = problem_->G0.transpose();
    kkt0.mat(1, 0) = problem_->G0;
    kkt0.mat(1, 1).diagonal().array() = -mudyn;
    kkt0.rhs[0] = -ps[0];
    kkt0.rhs[1] = -problem_->g0;
    kkt0.ldl.compute(kkt0.mat.matrix());
    kkt0.ldl.solveInPlace(kkt0.rhs.matrix());

    return true;
  }

  bool forward(std::vector<VectorXs> &xs, std::vector<VectorXs> &us,
               std::vector<VectorXs> &vs, std::vector<VectorXs> &lbdas,
               const std::optional<ConstVectorRef> & = std::nullopt) const {
    ALIGATOR_NOMALLOC_BEGIN;
    xs[0] = kkt0.rhs[0];
    lbdas[0] = kkt0.rhs[1];

    uint N = (uint)problem_->horizon();
    assert(xs.size() == N + 1);
    assert(vs.size() == N + 1);
    assert(lbdas.size() == N + 1);
    for (uint i = 0; i <= N; i++) {
      const FactorData &d = datas[i];
      ConstVectorRef kff = d.ff[0];
      ConstVectorRef zff = d.ff[1];
      ConstVectorRef lff = d.ff[2];
      ConstVectorRef yff = d.ff[3];

      ConstMatrixRef K = d.fb.blockRow(0);
      ConstMatrixRef Z = d.fb.blockRow(1);
      ConstMatrixRef Lfb = d.fb.blockRow(2);
      ConstMatrixRef Yfb = d.fb.blockRow(3);

      us[i].noalias() = kff + K * xs[i];
      vs[i].noalias() = zff + Z * xs[i];

      if (i == N)
        break;
      lbdas[i + 1].noalias() = lff + Lfb * xs[i];
      xs[i + 1].noalias() = yff + Yfb * xs[i];
    }
    ALIGATOR_NOMALLOC_END;
    return true;
  }
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template class RiccatiSolverDense<context::Scalar>;
#endif

} // namespace aligator::gar
