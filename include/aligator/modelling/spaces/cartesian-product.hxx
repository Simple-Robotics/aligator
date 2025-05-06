#pragma once

#include "aligator/modelling/spaces/cartesian-product.hpp"

namespace aligator {

#define ALIGATOR_CHECK_VEC_SIZE(v, n)                                          \
  if (v.size() != n)                                                           \
  ALIGATOR_DOMAIN_ERROR("Invalid size for vector (expected {:d}, got {:d})",   \
                        v.size(), n)

template <typename Scalar>
void CartesianProductTpl<Scalar>::neutral_impl(VectorRef out) const {
  Eigen::Index c = 0;
  for (std::size_t i = 0; i < numComponents(); i++) {
    const long n = m_components[i]->nx();
    m_components[i]->neutral(out.segment(c, n));
    c += n;
  }
}

template <typename Scalar>
void CartesianProductTpl<Scalar>::rand_impl(VectorRef out) const {
  Eigen::Index c = 0;
  for (std::size_t i = 0; i < numComponents(); i++) {
    const long n = m_components[i]->nx();
    m_components[i]->rand(out.segment(c, n));
    c += n;
  }
}

template <typename Scalar>
bool CartesianProductTpl<Scalar>::isNormalized(const ConstVectorRef &x) const {
  bool res = true;
  auto xs = this->split(x);
  for (std::size_t i = 0; i < numComponents(); i++) {
    res |= m_components[i]->isNormalized(xs[i]);
  }
  return res;
}

template <typename Scalar>
template <class VectorType, class U>
std::vector<U> CartesianProductTpl<Scalar>::split_impl(VectorType &x) const {
  ALIGATOR_CHECK_VEC_SIZE(x, this->nx());
  std::vector<U> out;
  Eigen::Index c = 0;
  for (std::size_t i = 0; i < numComponents(); i++) {
    const long n = m_components[i]->nx();
    out.push_back(x.segment(c, n));
    c += n;
  }
  return out;
}

template <typename Scalar>
template <class VectorType, class U>
std::vector<U>
CartesianProductTpl<Scalar>::split_vector_impl(VectorType &v) const {
  ALIGATOR_CHECK_VEC_SIZE(v, this->ndx());
  std::vector<U> out;
  Eigen::Index c = 0;
  for (std::size_t i = 0; i < numComponents(); i++) {
    const long n = m_components[i]->ndx();
    out.push_back(v.segment(c, n));
    c += n;
  }
  return out;
}

template <typename Scalar>
auto CartesianProductTpl<Scalar>::merge(const std::vector<VectorXs> &xs) const
    -> VectorXs {
  VectorXs out(this->nx());
  Eigen::Index c = 0;
  for (std::size_t i = 0; i < numComponents(); i++) {
    const long n = m_components[i]->nx();
    out.segment(c, n) = xs[i];
    c += n;
  }
  return out;
}

template <typename Scalar>
auto CartesianProductTpl<Scalar>::merge_vector(
    const std::vector<VectorXs> &vs) const -> VectorXs {
  VectorXs out(this->ndx());
  Eigen::Index c = 0;
  for (std::size_t i = 0; i < numComponents(); i++) {
    const long n = m_components[i]->ndx();
    out.segment(c, n) = vs[i];
    c += n;
  }
  return out;
}

template <typename Scalar>
void CartesianProductTpl<Scalar>::integrate_impl(const ConstVectorRef &x,
                                                 const ConstVectorRef &v,
                                                 VectorRef out) const {
  assert(nx() == out.size());
  Eigen::Index cq = 0, cv = 0;
  for (std::size_t i = 0; i < numComponents(); i++) {
    const long nq = getComponent(i).nx();
    const long nv = getComponent(i).ndx();
    auto sx = x.segment(cq, nq);
    auto sv = v.segment(cv, nv);

    auto sout = out.segment(cq, nq);

    getComponent(i).integrate(sx, sv, sout);
    cq += nq;
    cv += nv;
  }
}

template <typename Scalar>
void CartesianProductTpl<Scalar>::difference_impl(const ConstVectorRef &x0,
                                                  const ConstVectorRef &x1,
                                                  VectorRef out) const {
  assert(ndx() == out.size());
  Eigen::Index cq = 0, cv = 0;
  for (std::size_t i = 0; i < numComponents(); i++) {
    const long nq = getComponent(i).nx();
    const long nv = getComponent(i).ndx();
    auto sx0 = x0.segment(cq, nq);
    auto sx1 = x1.segment(cq, nq);

    auto sout = out.segment(cv, nv);

    getComponent(i).difference(sx0, sx1, sout);
    cq += nq;
    cv += nv;
  }
}

template <typename Scalar>
void CartesianProductTpl<Scalar>::Jintegrate_impl(const ConstVectorRef &x,
                                                  const ConstVectorRef &v,
                                                  MatrixRef Jout,
                                                  int arg) const {
  assert(ndx() == Jout.rows());
  Jout.setZero();
  Eigen::Index cq = 0, cv = 0;
  for (std::size_t i = 0; i < numComponents(); i++) {
    const long nq = getComponent(i).nx();
    const long nv = getComponent(i).ndx();
    auto sx = x.segment(cq, nq);
    auto sv = v.segment(cv, nv);

    auto sJout = Jout.block(cv, cv, nv, nv);

    getComponent(i).Jintegrate(sx, sv, sJout, arg);
    cq += nq;
    cv += nv;
  }
}

template <typename Scalar>
void CartesianProductTpl<Scalar>::JintegrateTransport_impl(
    const ConstVectorRef &x, const ConstVectorRef &v, MatrixRef Jout,
    int arg) const {
  Eigen::Index cq = 0, cv = 0;
  for (std::size_t i = 0; i < numComponents(); i++) {
    const long nq = m_components[i]->nx();
    auto sx = x.segment(cq, nq);
    cq += nq;

    const long nv = m_components[i]->ndx();
    auto sv = v.segment(cv, nv);
    auto sJout = Jout.middleRows(cv, nv);
    cv += nv;

    getComponent(i).JintegrateTransport(sx, sv, sJout, arg);
  }
}

template <typename Scalar>
void CartesianProductTpl<Scalar>::Jdifference_impl(const ConstVectorRef &x0,
                                                   const ConstVectorRef &x1,
                                                   MatrixRef Jout,
                                                   int arg) const {
  assert(ndx() == Jout.rows());
  Jout.setZero();
  Eigen::Index cq = 0, cv = 0;
  for (std::size_t i = 0; i < numComponents(); i++) {
    const long nq = m_components[i]->nx();
    auto sx0 = x0.segment(cq, nq);
    auto sx1 = x1.segment(cq, nq);
    cq += nq;

    const long nv = m_components[i]->ndx();
    auto sJout = Jout.block(cv, cv, nv, nv);
    cv += nv;

    getComponent(i).Jdifference(sx0, sx1, sJout, arg);
  }
}

} // namespace aligator
