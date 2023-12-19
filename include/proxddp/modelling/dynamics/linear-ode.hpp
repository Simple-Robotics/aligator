#pragma once

#include "proxddp/modelling/dynamics/ode-abstract.hpp"
#include <proxsuite-nlp/modelling/spaces/vector-space.hpp>

namespace proxddp {
namespace dynamics {
/**
 * @brief   Linear ordinary differential equation \f$\dot{x} = Ax + Bu\f$.
 *
 * @details This equation may be defined over a manifold's tangent space.
 */
template <typename _Scalar> struct LinearODETpl : ODEAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  PROXDDP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ODEAbstractTpl<Scalar>;
  using ODEData = ODEDataTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using ManifoldPtr = shared_ptr<Manifold>;

  MatrixXs A_, B_;
  VectorXs c_;

  /// @brief Standard constructor with state space, matrices \f$A,B\f$ and
  /// constant term \f$c\f$.
  LinearODETpl(const ManifoldPtr &space, const MatrixXs &A, const MatrixXs &B,
               const VectorXs &c)
      : Base(space, (int)B.cols()), A_(A), B_(B), c_(c) {
    if (A.cols() != space->ndx()) {
      PROXDDP_DOMAIN_ERROR(fmt::format(
          "A.cols() should be equal to space.ndx()! (got {:d} and {:d})",
          A.cols(), space->ndx()));
    }
    bool rows_ok = (A.rows() == space->ndx()) && (B.rows() == space->ndx()) &&
                   (c.rows() == space->ndx());
    if (!rows_ok) {
      PROXDDP_DOMAIN_ERROR("Input matrices have wrong number of rows.");
    }
  }

  /**
   * Constructor matrices \f$A,B\f$ and constant term \f$c\f$.
   * The state space is inferred to be a vector space.
   */
  LinearODETpl(const MatrixXs &A, const MatrixXs &B, const VectorXs &c)
      : LinearODETpl(std::make_shared<proxsuite::nlp::VectorSpaceTpl<Scalar>>(
                         (int)A.rows()),
                     A, B, c) {}

  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               ODEData &data) const;
  void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                ODEData &data) const;
  virtual shared_ptr<ContinuousDynamicsDataTpl<Scalar>> createData() const {
    auto data = Base::createData();
    data->Jx_ = A_;
    data->Ju_ = B_;
    return data;
  }
};

} // namespace dynamics
} // namespace proxddp

#include "proxddp/modelling/dynamics/linear-ode.hxx"
