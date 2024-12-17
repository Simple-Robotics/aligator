#pragma once

#include "lqr-problem.hpp"
#include "aligator/memory/eigen-map.hpp"

namespace aligator::gar {

template <typename Scalar>
LqrKnotTpl<Scalar>::LqrKnotTpl(no_alloc_t, allocator_type alloc)
    : Q(NULL, 0, 0), S(NULL, 0, 0), R(NULL, 0, 0), q(NULL, 0), r(NULL, 0), //
      A(NULL, 0, 0), B(NULL, 0, 0), E(NULL, 0, 0), f(NULL, 0),             //
      C(NULL, 0, 0), D(NULL, 0, 0), d(NULL, 0),                            //
      Gth(NULL, 0, 0), Gx(NULL, 0, 0), Gu(NULL, 0, 0), Gv(NULL, 0, 0),
      gamma(NULL, 0), m_empty_after_move(true), m_allocator(alloc) {}

template <typename Scalar>
LqrKnotTpl<Scalar>::LqrKnotTpl(uint nx, uint nu, uint nc, uint nx2, uint nth,
                               allocator_type alloc)
    : nx(nx), nu(nu), nc(nc), nx2(nx2), nth(nth), //
      Q(allocate_eigen_map<MatrixXs>(alloc, nx, nx)),
      S(allocate_eigen_map<MatrixXs>(alloc, nx, nu)),
      R(allocate_eigen_map<MatrixXs>(alloc, nu, nu)),
      q(allocate_eigen_map<VectorXs>(alloc, nx)),
      r(allocate_eigen_map<VectorXs>(alloc, nu)),
      A(allocate_eigen_map<MatrixXs>(alloc, nx2, nx)),
      B(allocate_eigen_map<MatrixXs>(alloc, nx2, nu)),
      E(allocate_eigen_map<MatrixXs>(alloc, nx2, nx2)),
      f(allocate_eigen_map<VectorXs>(alloc, nx2)),
      C(allocate_eigen_map<MatrixXs>(alloc, nc, nx)),
      D(allocate_eigen_map<MatrixXs>(alloc, nc, nu)),
      d(allocate_eigen_map<VectorXs>(alloc, nc)),
      Gth(allocate_eigen_map<MatrixXs>(alloc, nth, nth)),
      Gx(allocate_eigen_map<MatrixXs>(alloc, nx, nth)),
      Gu(allocate_eigen_map<MatrixXs>(alloc, nu, nth)),
      Gv(allocate_eigen_map<MatrixXs>(alloc, nc, nth)),
      gamma(allocate_eigen_map<VectorXs>(alloc, nth)),
      m_empty_after_move(false), m_allocator(std::move(alloc)) {}

template <typename Scalar> LqrKnotTpl<Scalar>::~LqrKnotTpl() {
  if (!m_empty_after_move)
    this->deallocate();
}

template <typename Scalar> void LqrKnotTpl<Scalar>::deallocate() {
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

template <typename Scalar>
void LqrKnotTpl<Scalar>::assign(const LqrKnotTpl &other) {
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
LqrKnotTpl<Scalar>::LqrKnotTpl(const LqrKnotTpl &other, allocator_type alloc)
    : LqrKnotTpl(other.nx, other.nu, other.nc, other.nx2, other.nth, alloc) {
  this->assign(other);
  assert(!m_empty_after_move);
}

#define _c(name) name(std::move(other.name))
template <typename Scalar>
LqrKnotTpl<Scalar>::LqrKnotTpl(LqrKnotTpl &&other)
    : nx(other.nx), nu(other.nu), nc(other.nc), nx2(other.nx2),
      nth(other.nth),                             //
      _c(Q), _c(S), _c(R), _c(q), _c(r),          //
      _c(A), _c(B), _c(E), _c(f),                 //
      _c(C), _c(D), _c(d),                        //
      _c(Gth), _c(Gx), _c(Gu), _c(Gv), _c(gamma), //
      m_empty_after_move(false), m_allocator(std::move(other.m_allocator)) {
  other.m_empty_after_move = true;
}
#undef _c

template <typename Scalar>
LqrKnotTpl<Scalar> &LqrKnotTpl<Scalar>::operator=(const LqrKnotTpl &other) {
  const bool same_dim = lqrKnotsSameDim(*this, other);
  if (same_dim) {
    // if dimensions compatible, do not reallocate memory.
    this->assign(other);
  } else {
    assert(!other.empty_after_move() && "Other should not be empty");
    // Allow allocation. Will use current allocator.
    // step 1: check if we are initialized. if so, deallocate.
    if (!m_empty_after_move)
      this->deallocate();

    // step 2: reallocate copies into the maps
#define _c(name) emplace_map_copy(name, other.name, m_allocator) // copy macro
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
    m_empty_after_move = false;
  }
  this->nx = other.nx;
  this->nu = other.nu;
  this->nc = other.nc;
  this->nx2 = other.nx2;
  this->nth = other.nth;
  return *this;
}

template <typename Scalar>
LqrKnotTpl<Scalar> &LqrKnotTpl<Scalar>::operator=(LqrKnotTpl &&other) {
  using allocator_traits = std::allocator_traits<allocator_type>;
  // for polymorphic_allocator types, the allocator is NOT moved on container
  // move assignment.
  assert(!allocator_traits::propagate_on_container_move_assignment::value);
  this->nx = other.nx;
  this->nu = other.nu;
  this->nc = other.nc;
  this->nx2 = other.nx2;
  this->nth = other.nth;

  // check if allocators are the same e.g. ALLOCATE FROM THE SAME RESOURCE
  // operator== on polymorphic allocators check address and state of underlying
  // memory resource.
  if (m_allocator == other.m_allocator) {
    // steal data from the other maps.
    // this is correct because the allocator for 'this'
    // which will clean them up *is* the allocator for 'other'.
#define _c(name) emplace_map_steal(name, other.name)
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
  } else {
    // otherwise, deallocate if necessary, allocate-copy data from other
    if (!m_empty_after_move)
      this->deallocate();

#define _c(name) emplace_map_copy(name, other.name, m_allocator) // copy macro
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

  other.m_empty_after_move = true;
  // reset flag if necessary
  m_empty_after_move = false;
  return *this;
}

template <typename Scalar>
LqrKnotTpl<Scalar> &LqrKnotTpl<Scalar>::addParameterization(uint nth) {
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
bool LqrKnotTpl<Scalar>::isApprox(const LqrKnotTpl &other, Scalar prec) const {
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
LQRProblemTpl<Scalar>::LQRProblemTpl(const KnotVector &knots, long nc0)
    : stages(knots, knots.get_allocator()),
      G0(allocate_eigen_map<MatrixXs>(get_allocator(), nc0,
                                      knots.empty() ? 0 : knots[0].nx)),
      g0(allocate_eigen_map<VectorXs>(get_allocator(), nc0)),
      m_is_invalid(false) {}

template <typename Scalar>
LQRProblemTpl<Scalar>::LQRProblemTpl(KnotVector &&knots, long nc0)
    : stages(knots, knots.get_allocator()),
      G0(allocate_eigen_map<MatrixXs>(get_allocator(), nc0,
                                      knots.empty() ? 0 : knots[0].nx)),
      g0(allocate_eigen_map<VectorXs>(get_allocator(), nc0)),
      m_is_invalid(false) {}

template <typename Scalar>
LQRProblemTpl<Scalar>::LQRProblemTpl(LQRProblemTpl &&other)
    : stages(std::move(other.stages)), G0(std::move(other.G0)),
      g0(std::move(other.g0)), m_is_invalid(false) {
  other.m_is_invalid = true;
}

template <typename Scalar> LQRProblemTpl<Scalar>::~LQRProblemTpl() {
  if (!m_is_invalid) {
    deallocate_map(G0, get_allocator());
    deallocate_map(g0, get_allocator());
  }
}

template <typename Scalar>
Scalar LQRProblemTpl<Scalar>::evaluate(
    const VectorOfVectors &xs, const VectorOfVectors &us,
    const std::optional<ConstVectorRef> &theta_) const {
  if (xs.size() != horizon() + 1)
    return 0.;
  if (us.size() < horizon())
    return 0.;

  if (stages.empty())
    return 0.;

  Scalar ret = 0.;
  const auto N = uint(horizon());
  for (uint i = 0; i <= N; i++) {
    const LqrKnotTpl<Scalar> &knot = stages[i];
    ret += 0.5 * xs[i].dot(knot.Q * xs[i]) + xs[i].dot(knot.q);
    if (i == N)
      break;
    ret += 0.5 * us[i].dot(knot.R * us[i]) + us[i].dot(knot.r);
    ret += xs[i].dot(knot.S * us[i]);
  }

  if (!isParameterized())
    return ret;

  if (theta_.has_value()) {
    ConstVectorRef th = theta_.value();
    for (uint i = 0; i <= N; i++) {
      const KnotType &knot = stages[i];
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
