#pragma once

#include "aligator/context.hpp"
#include <proxsuite-nlp/manifold-base.hpp>

#include <crocoddyl/core/state-base.hpp>
#include <boost/shared_ptr.hpp>

namespace aligator::compat::croc {

/// @brief Wraps a crocoddyl::StateAbstractTpl to a manifold
/// (aligator::ManifoldAbstractTpl).
template <typename _Scalar>
struct StateWrapperTpl : ManifoldAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using PointType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using TangentVectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  using StateAbstract = crocoddyl::StateAbstractTpl<Scalar>;

  shared_ptr<StateAbstract> croc_state;

  explicit StateWrapperTpl(const shared_ptr<StateAbstract> &state)
      : croc_state(state) {}

  int nx() const { return (int)croc_state->get_nx(); }
  int ndx() const { return (int)croc_state->get_ndx(); }

  PointType neutral() const { return croc_state->zero(); }
  PointType rand() const { return croc_state->rand(); }

  void integrate_impl(const ConstVectorRef &x, const ConstVectorRef &v,
                      VectorRef out) const {
    croc_state->integrate(x, v, out);
  }

  void Jintegrate_impl(const ConstVectorRef &x, const ConstVectorRef &v,
                       MatrixRef Jout, int arg) const {
    croc_state->Jintegrate(x, v, Jout, Jout, convert_to_firstsecond(arg));
  }

  void difference_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                       VectorRef out) const {
    croc_state->diff(x0, x1, out);
  }

  void Jdifference_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                        MatrixRef Jout, int arg) const {
    croc_state->Jdiff(x0, x1, Jout, Jout, convert_to_firstsecond(arg));
  }

  void JintegrateTransport(const ConstVectorRef &x, const ConstVectorRef &v,
                           MatrixRef Jout, int arg) const {
    croc_state->JintegrateTransport(x, v, Jout, convert_to_firstsecond(arg));
  }

  static crocoddyl::Jcomponent convert_to_firstsecond(int arg) {
    if (arg == 0) {
      return crocoddyl::first;
    } else if (arg == 1) {
      return crocoddyl::second;
    } else {
      return crocoddyl::both;
    }
  }
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct StateWrapperTpl<context::Scalar>;
#endif

} // namespace aligator::compat::croc
