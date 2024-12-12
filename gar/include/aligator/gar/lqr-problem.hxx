#pragma once

#include "lqr-problem.hpp"

namespace aligator::gar {

namespace detail {
template <typename EigenType, int MapOptions,
          typename Scalar = typename EigenType::Scalar>
void emplaceMap(Eigen::Map<EigenType, MapOptions> &map, long size,
                Scalar *ptr) {
  using MapType = Eigen::Map<EigenType, MapOptions>;
  new (&map) MapType{ptr, size};
}

/// \brief Placement-new a map type using the provided memory pointer.
template <typename EigenType, int MapOptions,
          typename Scalar = typename EigenType::Scalar>
void emplaceMap(Eigen::Map<EigenType, MapOptions> &map, long rows, long cols,
                Scalar *ptr) {
  using MapType = Eigen::Map<EigenType, MapOptions>;
  new (&map) MapType{ptr, rows, cols};
}
} // namespace detail

template <typename Scalar>
LQRKnotTpl<Scalar>::LQRKnotTpl(no_alloc_t, uint nx, uint nu, uint nc, uint nx2,
                               uint nth)
    : nx(nx), nu(nu), nc(nc), nx2(nx2), nth(nth),                          //
      Q(NULL, 0, 0), S(NULL, 0, 0), R(NULL, 0, 0), q(NULL, 0), r(NULL, 0), //
      A(NULL, 0, 0), B(NULL, 0, 0), E(NULL, 0, 0), f(NULL, 0),             //
      C(NULL, 0, 0), D(NULL, 0, 0), d(NULL, 0),                            //
      Gth(NULL, 0, 0), Gx(NULL, 0, 0), Gu(NULL, 0, 0), Gv(NULL, 0, 0),
      gamma(NULL, 0), //
      memory(NULL), req(Alignment) {}

template <typename Scalar>
LQRKnotTpl<Scalar>::LQRKnotTpl(uint nx, uint nu, uint nc, uint nx2, uint nth)
    : LQRKnotTpl(no_alloc, nx, nu, nc, nx2, nth) {

  this->allocate();
  this->initialize();
}

template <typename Scalar> void LQRKnotTpl<Scalar>::allocate() {
  req.addArray<double>(nx, nx)    // Q
      .addArray<double>(nx, nu)   // S
      .addArray<double>(nu, nu)   // R
      .addArray<double>(nx)       // q
      .addArray<double>(nu)       // r
      .addArray<double>(nx2, nx)  // A
      .addArray<double>(nx2, nu)  // B
      .addArray<double>(nx2, nx2) // E
      .addArray<double>(nx2)      // f
      .addArray<double>(nc, nx)   // C
      .addArray<double>(nc, nu)   // D
      .addArray<double>(nc)       // d
      .addArray<double>(nth, nth) // Gth
      .addArray<double>(nx, nth)  // Gx
      .addArray<double>(nu, nth)  // Gu
      .addArray<double>(nc, nth)  // Gv
      .addArray<double>(nth);     // gamma

  this->memory = static_cast<Scalar *>(req.allocate());
  std::memset(memory, 0, req.totalBytes());
}

template <typename Scalar> void LQRKnotTpl<Scalar>::initialize() {
  Scalar *ptr = memory;
  detail::emplaceMap(Q, nx, nx, ptr);
  req.advance(ptr);
  detail::emplaceMap(S, nx, nu, ptr);
  req.advance(ptr);
  detail::emplaceMap(R, nu, nu, ptr);
  req.advance(ptr);
  detail::emplaceMap(q, nx, ptr);
  req.advance(ptr);
  detail::emplaceMap(r, nu, ptr);
  req.advance(ptr);

  detail::emplaceMap(A, nx2, nx, ptr);
  req.advance(ptr);
  detail::emplaceMap(B, nx2, nu, ptr);
  req.advance(ptr);
  detail::emplaceMap(E, nx2, nx2, ptr);
  req.advance(ptr);
  detail::emplaceMap(f, nx2, ptr);
  req.advance(ptr);

  detail::emplaceMap(C, nc, nx, ptr);
  req.advance(ptr);
  detail::emplaceMap(D, nc, nu, ptr);
  req.advance(ptr);
  detail::emplaceMap(d, nc, ptr);
  req.advance(ptr);

  detail::emplaceMap(Gth, nth, nth, ptr);
  req.advance(ptr);
  detail::emplaceMap(Gx, nx, nth, ptr);
  req.advance(ptr);
  detail::emplaceMap(Gu, nu, nth, ptr);
  req.advance(ptr);
  detail::emplaceMap(Gv, nc, nth, ptr);
  req.advance(ptr);
  detail::emplaceMap(gamma, nth, ptr);
  req.advance(ptr);

  req.reset();
}

template <typename Scalar>
void LQRKnotTpl<Scalar>::addParameterization(uint nth) {
  LQRKnotTpl copy(nx, nu, nc, nx2, nth);
  copy.Q = Q;
  copy.S = S;
  copy.R = R;
  copy.q = q;
  copy.r = r;

  copy.A = A;
  copy.B = B;
  copy.E = E;
  copy.f = f;

  copy.C = C;
  copy.D = D;
  copy.d = d;

  *this = LQRKnotTpl{copy};
}

template <typename Scalar>
LQRKnotTpl<Scalar>::LQRKnotTpl(const LQRKnotTpl &other)
    : LQRKnotTpl(no_alloc, other.nx, other.nu, other.nc, other.nx2, other.nth) {
  this->allocate();
  assert(req.totalBytes() == other.req.totalBytes());
  // copy memory over from other
  std::memcpy(memory, other.memory, other.req.totalBytes());
  this->initialize();
}

template <typename Scalar>
LQRKnotTpl<Scalar>::LQRKnotTpl(LQRKnotTpl &&other)
    : LQRKnotTpl(no_alloc, other.nx, other.nu, other.nc, other.nx2, other.nth) {
  // no need to allocate, just bring in the other
  // memory buffer
  memory = other.memory;
  other.memory = NULL;
  req = other.req;
  this->initialize();
}

template <typename Scalar>
LQRKnotTpl<Scalar> &LQRKnotTpl<Scalar>::operator=(const LQRKnotTpl &other) {
  this->~LQRKnotTpl();
  new (this) LQRKnotTpl{other};
  return *this;
}

template <typename Scalar>
LQRKnotTpl<Scalar> &LQRKnotTpl<Scalar>::operator=(LQRKnotTpl &&other) {
  swap(*this, other);
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
