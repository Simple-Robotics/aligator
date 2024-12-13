#pragma once

#include "lqr-problem.hpp"
#include "eigen-map-management.hpp"

namespace aligator::gar {

template <typename Scalar>
LQRKnotTpl<Scalar>::LQRKnotTpl(no_alloc_t, uint nx, uint nu, uint nc, uint nx2,
                               uint nth, const allocator_type &alloc)
    : nx(nx), nu(nu), nc(nc), nx2(nx2), nth(nth),                          //
      Q(NULL, 0, 0), S(NULL, 0, 0), R(NULL, 0, 0), q(NULL, 0), r(NULL, 0), //
      A(NULL, 0, 0), B(NULL, 0, 0), E(NULL, 0, 0), f(NULL, 0),             //
      C(NULL, 0, 0), D(NULL, 0, 0), d(NULL, 0),                            //
      Gth(NULL, 0, 0), Gx(NULL, 0, 0), Gu(NULL, 0, 0), Gv(NULL, 0, 0),
      gamma(NULL, 0), m_empty_after_move(true), m_allocator(alloc) {}
template <typename Scalar>
LQRKnotTpl<Scalar>::LQRKnotTpl(uint nx, uint nu, uint nc, uint nx2, uint nth,
                               const allocator_type &alloc)
    : LQRKnotTpl(no_alloc, nx, nu, nc, nx2, nth, alloc) {
  emplace_allocated_map(Q, nx, nx, m_allocator);
  emplace_allocated_map(S, nx, nu, m_allocator);
  emplace_allocated_map(R, nu, nu, m_allocator);
  emplace_allocated_map(q, nx, m_allocator);
  emplace_allocated_map(r, nu, m_allocator);

  emplace_allocated_map(A, nx2, nx, m_allocator);
  emplace_allocated_map(B, nx2, nu, m_allocator);
  emplace_allocated_map(E, nx2, nx2, m_allocator);
  emplace_allocated_map(f, nx2, m_allocator);

  emplace_allocated_map(C, nc, nx, m_allocator);
  emplace_allocated_map(D, nc, nu, m_allocator);
  emplace_allocated_map(d, nc, m_allocator);

  emplace_allocated_map(Gth, nth, nth, m_allocator);
  emplace_allocated_map(Gx, nx, nth, m_allocator);
  emplace_allocated_map(Gu, nu, nth, m_allocator);
  emplace_allocated_map(Gv, nc, nth, m_allocator);
  emplace_allocated_map(gamma, nth, m_allocator);
}

template <typename Scalar> LQRKnotTpl<Scalar>::~LQRKnotTpl() {
  if (!m_empty_after_move) {
    deallocate_map(Q, m_allocator);
    deallocate_map(S, m_allocator);
    deallocate_map(R, m_allocator);
    deallocate_map(q, m_allocator);
    deallocate_map(r, m_allocator);

    deallocate_map(A, m_allocator);
    deallocate_map(B, m_allocator);
    deallocate_map(E, m_allocator);
    deallocate_map(f, m_allocator);

    deallocate_map(C, m_allocator);
    deallocate_map(D, m_allocator);
    deallocate_map(d, m_allocator);

    deallocate_map(Gth, m_allocator);
    deallocate_map(Gx, m_allocator);
    deallocate_map(Gu, m_allocator);
    deallocate_map(Gv, m_allocator);
    deallocate_map(gamma, m_allocator);
  }
}

template <typename Scalar>
LQRKnotTpl<Scalar>::LQRKnotTpl(const LQRKnotTpl &other)
    : LQRKnotTpl(other.nx, other.nu, other.nc, other.nx2, other.nth,
                 other.get_allocator()) {
  this->Q = other.Q;
  this->S = other.S;
  this->R = other.R;
  this->q = other.q;
  this->r = other.r;

  this->A = other.A;
  this->B = other.B;
  this->E = other.E;
  this->f = other.f;

  this->C = other.C;
  this->D = other.D;
  this->d = other.d;

  this->Gth = other.Gth;
  this->Gx = other.Gx;
  this->Gu = other.Gu;
  this->Gv = other.Gv;
  this->gamma = other.gamma;
}

template <typename Scalar>
LQRKnotTpl<Scalar> &LQRKnotTpl<Scalar>::operator=(const LQRKnotTpl &other) {
  LQRKnotTpl copy{other};
  *this = std::move(copy);
  return *this;
}

template <typename Scalar>
LQRKnotTpl<Scalar> &LQRKnotTpl<Scalar>::operator=(LQRKnotTpl &&other) {
  this->nx = other.nx;
  this->nu = other.nu;
  this->nc = other.nc;
  this->nx2 = other.nx2;
  this->nth = other.nth;

  emplace_map_steal(this->Q, other.Q);
  emplace_map_steal(this->S, other.S);
  emplace_map_steal(this->R, other.R);
  emplace_map_steal(this->q, other.q);
  emplace_map_steal(this->r, other.r);

  emplace_map_steal(this->A, other.A);
  emplace_map_steal(this->B, other.B);
  emplace_map_steal(this->E, other.E);
  emplace_map_steal(this->f, other.f);

  emplace_map_steal(this->C, other.C);
  emplace_map_steal(this->D, other.D);
  emplace_map_steal(this->d, other.d);

  emplace_map_steal(this->Gth, other.Gth);
  emplace_map_steal(this->Gx, other.Gx);
  emplace_map_steal(this->Gu, other.Gu);
  emplace_map_steal(this->Gv, other.Gv);
  emplace_map_steal(this->gamma, other.gamma);

  other.m_empty_after_move = true;
  // reset flag if necessary
  m_empty_after_move = false;
  return *this;
}

template <typename Scalar>
LQRKnotTpl<Scalar> &LQRKnotTpl<Scalar>::addParameterization(uint nth) {
  this->nth = nth;
  deallocate_map(Gth, m_allocator);
  deallocate_map(Gx, m_allocator);
  deallocate_map(Gu, m_allocator);
  deallocate_map(Gv, m_allocator);
  deallocate_map(gamma, m_allocator);

  emplace_allocated_map(Gth, nth, nth, m_allocator);
  emplace_allocated_map(Gx, nx, nth, m_allocator);
  emplace_allocated_map(Gu, nx, nu, m_allocator);
  emplace_allocated_map(Gv, nx, nc, m_allocator);
  emplace_allocated_map(gamma, nth, m_allocator);
  return *this;
}

template <typename Scalar>
bool LQRKnotTpl<Scalar>::isApprox(const LQRKnotTpl &other, Scalar prec) const {
  bool cost = Q.isApprox(other.Q, prec) && S.isApprox(other.S, prec) &&
              R.isApprox(other.R, prec) && q.isApprox(other.q, prec) &&
              r.isApprox(other.r, prec);
  bool dyn = A.isApprox(other.A, prec) && B.isApprox(other.B, prec) &&
             E.isApprox(other.E, prec) && f.isApprox(other.f, prec);
  bool cstr = C.isApprox(other.C, prec) && D.isApprox(other.D, prec) &&
              d.isApprox(other.d, prec);
  bool th = Gth.isApprox(other.Gth, prec) && Gx.isApprox(other.Gx, prec) &&
            Gu.isApprox(other.Gu, prec) && Gv.isApprox(other.Gv, prec) &&
            gamma.isApprox(other.gamma, prec);
  return cost && dyn && cstr && th;
}

template <typename Scalar>
Scalar LQRProblemTpl<Scalar>::evaluate(
    const VectorOfVectors &xs, const VectorOfVectors &us,
    const std::optional<ConstVectorRef> &theta_) const {
  if ((int)xs.size() != horizon() + 1)
    return 0.;
  if ((int)us.size() < horizon())
    return 0.;

  if (!isInitialized())
    return 0.;

  Scalar ret = 0.;
  for (uint i = 0; i <= (uint)horizon(); i++) {
    const LQRKnotTpl<Scalar> &knot = stages[i];
    ret += 0.5 * xs[i].dot(knot.Q * xs[i]) + xs[i].dot(knot.q);
    if (i == (uint)horizon())
      break;
    ret += 0.5 * us[i].dot(knot.R * us[i]) + us[i].dot(knot.r);
    ret += xs[i].dot(knot.S * us[i]);
  }

  if (!isParameterized())
    return ret;

  if (theta_.has_value()) {
    ConstVectorRef th = theta_.value();
    for (uint i = 0; i <= (uint)horizon(); i++) {
      const LQRKnotTpl<Scalar> &knot = stages[i];
      ret += 0.5 * th.dot(knot.Gth * th);
      ret += th.dot(knot.Gx.transpose() * xs[i]);
      ret += th.dot(knot.gamma);
      if (i == (uint)horizon())
        break;
      ret += th.dot(knot.Gu.transpose() * us[i]);
    }
  }

  return ret;
}

} // namespace aligator::gar
