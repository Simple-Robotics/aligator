/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, 2023-2025 INRIA
#pragma once

#include "lqr-problem.hpp"

namespace aligator::gar {

template <typename Scalar>
LqrKnotTpl<Scalar>::LqrKnotTpl(no_alloc_t, allocator_type alloc)
    : Q(alloc), S(alloc), R(alloc), q(alloc), r(alloc), //
      A(alloc), B(alloc), E(alloc), f(alloc),           //
      C(alloc), D(alloc), d(alloc),                     //
      Gth(alloc), Gx(alloc), Gu(alloc), Gv(alloc), gamma(alloc),
      m_allocator(alloc) {}

template <typename Scalar>
LqrKnotTpl<Scalar>::LqrKnotTpl(uint nx, uint nu, uint nc, uint nx2, uint nth,
                               allocator_type alloc)
    : nx(nx), nu(nu), nc(nc), nx2(nx2), nth(nth), //
      Q(nx, nx, alloc), S(nx, nu, alloc), R(nu, nu, alloc), q(nx, alloc),
      r(nu, alloc), A(nx2, nx, alloc), B(nx2, nu, alloc), E(nx2, nx2, alloc),
      f(nx2, alloc), C(nc, nx, alloc), D(nc, nu, alloc), d(nc, alloc),
      Gth(nth, nth, alloc), Gx(nx, nth, alloc), Gu(nu, nth, alloc),
      Gv(nc, nth, alloc), gamma(nth, alloc), m_allocator(std::move(alloc)) {
  Q.setZero();
  S.setZero();
  R.setZero();
  q.setZero();
  r.setZero();

  A.setZero();
  B.setZero();
  E.setZero();
  f.setZero();

  C.setZero();
  D.setZero();
  d.setZero();

  Gth.setZero();
  Gx.setZero();
  Gu.setZero();
  Gv.setZero();
  gamma.setZero();
}

template <typename Scalar> LqrKnotTpl<Scalar>::~LqrKnotTpl() {}

template <typename Scalar>
void LqrKnotTpl<Scalar>::assign(const LqrKnotTpl &other) {
  this->nx = other.nx;
  this->nu = other.nu;
  this->nc = other.nc;
  this->nx2 = other.nx2;
  this->nth = other.nth;

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

#define _c(name) name(other.name, alloc)
template <typename Scalar>
LqrKnotTpl<Scalar>::LqrKnotTpl(const LqrKnotTpl &other, allocator_type alloc)
    : nx(other.nx), nu(other.nu), nc(other.nc), nx2(other.nx2),
      nth(other.nth),                             //
      _c(Q), _c(S), _c(R), _c(q), _c(r),          //
      _c(A), _c(B), _c(E), _c(f),                 //
      _c(C), _c(D), _c(d),                        //
      _c(Gth), _c(Gx), _c(Gu), _c(Gv), _c(gamma), //
      m_allocator(other.m_allocator) {}
#undef _c

#define _c(name) name(std::move(other.name))
template <typename Scalar>
LqrKnotTpl<Scalar>::LqrKnotTpl(LqrKnotTpl &&other)
    : nx(other.nx), nu(other.nu), nc(other.nc), nx2(other.nx2),
      nth(other.nth),                             //
      _c(Q), _c(S), _c(R), _c(q), _c(r),          //
      _c(A), _c(B), _c(E), _c(f),                 //
      _c(C), _c(D), _c(d),                        //
      _c(Gth), _c(Gx), _c(Gu), _c(Gv), _c(gamma), //
      m_allocator(other.m_allocator) {}
#undef _c

template <typename Scalar>
LqrKnotTpl<Scalar> &LqrKnotTpl<Scalar>::operator=(const LqrKnotTpl &other) {
  this->assign(other);
  return *this;
}

template <typename Scalar>
LqrKnotTpl<Scalar> &LqrKnotTpl<Scalar>::operator=(LqrKnotTpl &&other) {
#ifndef NDEBUG
  using allocator_traits = std::allocator_traits<allocator_type>;
  assert(!allocator_traits::propagate_on_container_move_assignment::value);
#endif
  // for polymorphic_allocator types, the allocator is NOT moved on container
  // move assignment.
  this->nx = other.nx;
  this->nu = other.nu;
  this->nc = other.nc;
  this->nx2 = other.nx2;
  this->nth = other.nth;

  {
#define _c(name) this->name = other.name
    _c(Q);
    _c(S);
    _c(R);
    _c(q);
    _c(r);

    _c(A);
    _c(B);
    _c(E);
    _c(f);

    _c(C);
    _c(D);
    _c(d);

    _c(Gth);
    _c(Gx);
    _c(Gu);
    _c(Gv);
    _c(gamma);
#undef _c
  }

  return *this;
}

template <typename Scalar>
LqrKnotTpl<Scalar> &LqrKnotTpl<Scalar>::addParameterization(uint nth) {
  this->nth = nth;
  Gth.setZero(nth, nth);
  Gx.setZero(nx, nth);
  Gu.setZero(nu, nth);
  Gv.setZero(nc, nth);
  gamma.setZero(nth);
  return *this;
}

template <typename Scalar>
bool LqrKnotTpl<Scalar>::isApprox(const LqrKnotTpl &other, Scalar prec) const {
  if (!lqrKnotsSameDim(*this, other))
    return false;
  if (!(Q.isApprox(other.Q, prec) && S.isApprox(other.S, prec) &&
        R.isApprox(other.R, prec) && q.isApprox(other.q, prec) &&
        r.isApprox(other.r, prec)))
    return false;

  if (!(A.isApprox(other.A, prec) && B.isApprox(other.B, prec) &&
        E.isApprox(other.E, prec) && f.isApprox(other.f, prec)))
    return false;

  if (!(C.isApprox(other.C, prec) && D.isApprox(other.D, prec) &&
        d.isApprox(other.d, prec)))
    return false;

  return Gth.isApprox(other.Gth, prec) && Gx.isApprox(other.Gx, prec) &&
         Gu.isApprox(other.Gu, prec) && Gv.isApprox(other.Gv, prec) &&
         gamma.isApprox(other.gamma, prec);
}

template <typename Scalar>
LqrProblemTpl<Scalar>::LqrProblemTpl(const KnotVector &knots, long nc0,
                                     allocator_type alloc)
    : G0(nc0, knots.empty() ? 0 : knots[0].nx, alloc), g0(nc0, alloc),
      stages(knots, alloc), m_is_invalid(false) {
  assert(check_allocators());
}

template <typename Scalar>
LqrProblemTpl<Scalar>::LqrProblemTpl(KnotVector &&knots, long nc0)
    : G0(nc0, knots.empty() ? 0 : knots[0].nx, knots.get_allocator()),
      g0(nc0, get_allocator()), stages(std::move(knots), get_allocator()),
      m_is_invalid(false) {
  assert(check_allocators());
}

template <typename Scalar> LqrProblemTpl<Scalar>::~LqrProblemTpl() {}

template <typename Scalar>
Scalar LqrProblemTpl<Scalar>::evaluate(
    const VectorOfVectors &xs, const VectorOfVectors &us,
    const std::optional<ConstVectorRef> &theta_) const {
  using const_view_t = typename LqrKnotTpl<Scalar>::const_view_t;

  if ((int)xs.size() != horizon() + 1)
    return 0.;
  if ((int)us.size() < horizon())
    return 0.;

  if (stages.empty())
    return 0.;

  Scalar ret = 0.;
  const auto N = uint(horizon());
  for (uint i = 0; i <= N; i++) {
    const_view_t knot = stages[i].to_const_view();
    ret += 0.5 * xs[i].dot(knot.Q * xs[i]) + xs[i].dot(knot.q);
    if (i == N)
      break;
    ret += 0.5 * us[i].dot(knot.R * us[i]) + us[i].dot(knot.r);
    ret += xs[i].dot(knot.S * us[i]);
  }

  if (isParameterized() && theta_.has_value()) {
    ConstVectorRef th = theta_.value();
    for (uint i = 0; i <= N; i++) {
      const_view_t knot = stages[i].to_const_view();
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
