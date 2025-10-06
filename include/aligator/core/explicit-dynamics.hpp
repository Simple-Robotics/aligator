/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include "aligator/context.hpp"
#include "aligator/core/manifold-base.hpp"
#include "aligator/third-party/polymorphic_cxx14.h"

#include <fmt/format.h>

namespace aligator {
using xyz::polymorphic;

/// @brief Explicit forward dynamics model \f$ x_{k+1} = f(x_k, u_k) \f$.
///
/// @details    Forward dynamics \f$ x_{k+1} = f(x_k, u_k) \f$.
///             The corresponding residuals for NLP-like formulations
//              are \f[ \bar{f}(x_k, u_k, x_{k+1}) = f(x_k, u_k) \ominus
//              x_{k+1}. \f]
///
template <typename _Scalar> struct ExplicitDynamicsModelTpl {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Data = ExplicitDynamicsDataTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;

  static constexpr bool is_explicit = true;

  /// Constructor requires providing the next state's manifold.
  ExplicitDynamicsModelTpl(const polymorphic<Manifold> &space, const int nu)
      : space_(space)
      , space_next_(space)
      , nu(nu) {}

  const Manifold &space() const { return *space_; }
  const Manifold &space_next() const { return *space_next_; }

  int nx1() const { return space_->nx(); }
  int ndx1() const { return space_->ndx(); }
  int nx2() const { return space_next_->nx(); }
  int ndx2() const { return space_next_->ndx(); }

  /// @brief Evaluate the forward discrete dynamics.
  void virtual forward(const ConstVectorRef &x, const ConstVectorRef &u,
                       Data &data) const = 0;

  /// @brief Compute the Jacobians of the forward dynamics.
  void virtual dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                        Data &data) const = 0;

  virtual shared_ptr<Data> createData() const {
    return std::make_shared<Data>(*this);
  }

  virtual ~ExplicitDynamicsModelTpl() = default;

  polymorphic<Manifold> space_;
  polymorphic<Manifold> space_next_;
  int nu;
};

/// @brief    Specific data struct for explicit dynamics
/// ExplicitDynamicsModelTpl.
template <typename _Scalar> struct ExplicitDynamicsDataTpl {
protected:
  int ndx1, nu, ndx2;

public:
  using Scalar = _Scalar;
  using Model = ExplicitDynamicsModelTpl<Scalar>;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  /// Next state.
  VectorXs xnext_;
  // Jacobians
  MatrixXs jac_buffer_;
  MatrixXs Jtmp_xnext;

  auto Jx() { return jac_buffer_.leftCols(ndx1); }
  auto Jx() const { return jac_buffer_.leftCols(ndx1); }
  auto Ju() { return jac_buffer_.rightCols(nu); }
  auto Ju() const { return jac_buffer_.rightCols(nu); }

  ExplicitDynamicsDataTpl(const Model &model)
      : ndx1(model.ndx1())
      , nu(model.nu)
      , ndx2(model.ndx2())
      , xnext_(model.nx2())
      , jac_buffer_(ndx2, ndx1 + nu)
      , Jtmp_xnext(ndx2, ndx2) {
    xnext_ = model.space().neutral();
    jac_buffer_.setZero();
    Jtmp_xnext.setZero();
  }

  virtual ~ExplicitDynamicsDataTpl() = default;

  friend std::ostream &operator<<(std::ostream &oss,
                                  const ExplicitDynamicsDataTpl &self) {
    oss << "ExplicitDynamicsData { ";
    oss << fmt::format("ndx: {:d},  ", self.ndx1);
    oss << fmt::format("nu:  {:d}", self.nu);
    oss << " }";
    return oss;
  }
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct ExplicitDynamicsModelTpl<context::Scalar>;
extern template struct ExplicitDynamicsDataTpl<context::Scalar>;
#endif
} // namespace aligator
