#pragma once

#include "./riccati.hpp"

namespace aligator {
namespace gar {
template <typename Scalar>
bool ProximalRiccatiSolverBackward<Scalar>::run(Scalar mudyn, Scalar mueq) {
  if (horizon() < 0)
    return false;

  ALIGATOR_NOMALLOC_BEGIN;
  // terminal node
  {
    stage_solve_data_t &d = datas[horizon()];
    value_t &vc = d.vm;
    const knot_t &model = knots.back();
    // fill cost-to-go matrix
    VectorRef zff = d.ff.tail(model.nc);
    MatrixRef Z = d.fb.bottomRows(model.nc);

    auto Ct = model.C.transpose();

    Z.noalias() = model.C / mueq;
    zff.noalias() = model.d / mueq;

    vc.Pmat.noalias() = model.Q + Ct * Z;
    vc.pvec.noalias() = model.q + Ct * zff;
  }

  size_t t = size_t(horizon() - 1);
  while (true) {
    stage_solve_data_t &d = datas[t];
    value_t &vn = datas[t + 1].vm;
    const knot_t &model = knots[t];
    fmt::print("[bwd] t={:>2d}\n", t);

    // compute decomposition in-place
    vn.chol.compute(vn.Pmat);
    d.PinvEt = model.E.transpose();
    vn.chol.solveInPlace(d.PinvEt);
    d.wvec = vn.chol.solve(-vn.pvec);
    ALIGATOR_NOMALLOC_END;
    {
      auto pmatrecerr =
          math::infty_norm(vn.Pmat - vn.chol.reconstructedMatrix());
      auto pinverr = math::infty_norm(vn.Pmat * d.PinvEt - model.E.transpose());
      auto wvecerr = math::infty_norm(vn.Pmat * d.wvec + vn.pvec);
      fmt::print("Pmatrec = {:4.3e}\n", pmatrecerr);
      fmt::print("Pinverr = {:4.3e}\n", pinverr);
      fmt::print("wvecerr = {:4.3e}\n", wvecerr);
    }
    ALIGATOR_NOMALLOC_BEGIN;

    vn.Lbmat.noalias() = model.E * d.PinvEt;
    vn.Lbmat.diagonal().array() += mudyn;

    // compute decomposition in-place
    vn.chol.compute(vn.Lbmat);
    vn.Vmat.setIdentity();
    vn.chol.solveInPlace(vn.Vmat); // evaluate inverse of Lambda
    vn.vvec.noalias() = model.f + model.E * d.wvec;
    ALIGATOR_NOMALLOC_END;
    {
      auto lbmatrecerr =
          math::infty_norm(vn.Lbmat - vn.chol.reconstructedMatrix());
      fmt::print("recerr  = {:4.3e}\n", lbmatrecerr);
    }
    ALIGATOR_NOMALLOC_BEGIN;

    // fill in hamiltonian
    computeKktTerms(model, d, vn);

    // fill feedback system
    d.Mmat.R() = d.hmlt.Rhat;
    d.Mmat.D() = model.D;
    d.Mmat.dual().setConstant(-mueq);
    d.Mmat.data = d.Mmat.data.template selfadjointView<Eigen::Lower>();
    Eigen::LDLT<MatrixXs> &ldlt = d.Mmat.chol;
    ldlt.compute(d.Mmat.data);

    value_t &vc = d.vm;
    VectorRef kff = d.ff.head(model.nu);
    VectorRef zff = d.ff.tail(model.nc);
    kff = -d.hmlt.rhat;
    zff = -model.d;

    MatrixRef K = d.fb.topRows(model.nu);
    MatrixRef Z = d.fb.bottomRows(model.nc);
    K = -d.hmlt.Shat.transpose();
    Z = -model.C;
#ifdef NDEBUG
    ldlt.solveInPlace(d.ff);
    ldlt.solveInPlace(d.fb);
#else
    ALIGATOR_NOMALLOC_END;
    VectorXs sff = ldlt.solve(d.ff);
    MatrixXs sfb = ldlt.solve(d.fb);
    {
      fmt::print("reduced KKT system:\n{}\n", d.Mmat.data);
      fmt::print("KKT rhs:\n{}\n{}\n", d.ff, d.fb);
      auto fferr = math::infty_norm(d.Mmat.data * sff + d.ff);
      auto fberr = math::infty_norm(d.Mmat.data * sfb + d.fb);
      fmt::print("ff sol.\n{}\n", sff);
      fmt::print("fb sol.\n{}\n", sfb);
      fmt::print("fferr = {:4.3e}\n", fferr);
      fmt::print("fberr = {:4.3e}\n", fberr);
      auto ldltErr = math::infty_norm(d.Mmat.data - ldlt.reconstructedMatrix());
      fmt::print("ldlterr = {:4.3e}\n", ldltErr);
    }
    ALIGATOR_NOMALLOC_BEGIN;
    d.ff = sff;
    d.fb = sfb;
#endif

    auto Ct = model.C.transpose();
    vc.Pmat.noalias() = d.hmlt.Qhat + model.S * K + Ct * Z;
    vc.pvec.noalias() = d.hmlt.qhat + model.S * kff + Ct * zff;

    if (t == 0)
      break;
    --t;
  }

  stage_solve_data_t &d0 = datas[0];
  value_t &vinit = d0.vm;
  vinit.Vmat = vinit.Pmat;
  vinit.vvec = vinit.pvec;
  vinit.chol.compute(vinit.Pmat);

  ALIGATOR_NOMALLOC_END;
  {
    auto P0err =
        math::infty_norm(vinit.Pmat - vinit.chol.reconstructedMatrix());
    fmt::print("P0err = {:4.3e}\n", P0err);
    fmt::print("P0 =\n{}\n", vinit.Pmat);
    fmt::print("p0 =\n{}\n", vinit.pvec);
  }

  return true;
}

template <typename Scalar>
void ProximalRiccatiSolverBackward<Scalar>::computeKktTerms(
    const knot_t &model, stage_solve_data_t &d, const value_t &vnext) {
  hmlt_t &hmlt = d.hmlt;
  hmlt.AtV.noalias() = model.A.transpose() * vnext.Vmat;
  hmlt.BtV.noalias() = model.B.transpose() * vnext.Vmat;

  hmlt.Qhat.noalias() = model.Q + hmlt.AtV * model.A;
  hmlt.Rhat.noalias() = model.R + hmlt.BtV * model.B;
  hmlt.Shat.noalias() = model.S + hmlt.AtV * model.B;

  hmlt.qhat.noalias() = model.q + hmlt.AtV * vnext.vvec;
  hmlt.rhat.noalias() = model.r + hmlt.BtV * vnext.vvec;
}

template <typename Scalar>
bool ProximalRiccatiSolverForward<Scalar>::run(bwd_algo_t &bwd, vecvec_t &xs,
                                               vecvec_t &us, vecvec_t &vs,
                                               vecvec_t &lbdas) {
  using stage_solve_data_t = typename bwd_algo_t::stage_solve_data_t;

  const std::vector<knot_t> &knots = bwd.knots;

  // solve initial stage
  {
    stage_solve_data_t &d0 = bwd.datas[0];
    xs[0] = d0.vm.chol.solve(-d0.vm.pvec);
    auto err = math::infty_norm(d0.vm.pvec + d0.vm.Pmat * xs[0]);
    fmt::print("err[x0] = {:4.3e}\n", err);
  }
  ALIGATOR_NOMALLOC_BEGIN;

  size_t N = (size_t)bwd.horizon();
  for (size_t t = 0; t <= N; t++) {
    stage_solve_data_t &d = bwd.datas[t];
    typename bwd_algo_t::value_t &vnext = bwd.datas[t + 1].vm;
    const knot_t &model = knots[t];

    MatrixRef K = d.fb.topRows(model.nu);    // control feedback
    MatrixRef Z = d.fb.bottomRows(model.nc); // multiplier feedback
    VectorRef kff = d.ff.head(model.nu);
    VectorRef zff = d.ff.tail(model.nc);

    vs[t].noalias() = zff + Z * xs[t];

    if (t == N)
      break;

    us[t].noalias() = kff + K * xs[t];
    // next costate
    // use xnext as a tmp buffer
    xs[t + 1].noalias() = vnext.vvec + model.A * xs[t] + model.B * us[t];
    lbdas[t].noalias() = vnext.Vmat * xs[t + 1];

    auto Wmat = -d.PinvEt;
    xs[t + 1].noalias() = d.wvec + Wmat * lbdas[t];
  }

  ALIGATOR_NOMALLOC_END;
  return true;
}

} // namespace gar
} // namespace aligator
