/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include "aligator/fwd.hpp"

namespace aligator {

/// @brief Base class for manifolds, to use in cost funcs, solvers...
///
template <typename _Scalar> struct ManifoldAbstractTpl {
public:
  using Scalar = _Scalar; /// Scalar type
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  /// Typedef for the tangent space, as a manifold.
  using TangentSpaceType = VectorSpaceTpl<Scalar, Eigen::Dynamic>;

  ManifoldAbstractTpl(int nx, int ndx)
      : nx_(nx)
      , ndx_(ndx) {}

  virtual ~ManifoldAbstractTpl() = default;

  /// @brief    Get manifold representation dimension.
  inline int nx() const { return nx_; }
  /// @brief    Get manifold tangent space dimension.
  inline int ndx() const { return ndx_; }

  /// @brief Get the neutral element \f$e \in M\f$ from the manifold (if this
  /// makes sense).
  [[nodiscard]] VectorXs neutral() const {
    VectorXs out(nx());
    neutral_impl(out);
    return out;
  }

  /// @copybrief neutral().
  void neutral(VectorRef out) const { neutral_impl(out); }

  /// @brief Sample a random point \f$x \in M\f$ on the manifold.
  [[nodiscard]] VectorXs rand() const {
    VectorXs out(nx());
    rand_impl(out);
    return out;
  }

  /// @copybrief rand().
  void rand(VectorRef out) const { rand_impl(out); }

  /// @brief Check if the input vector @p x is a viable element of the
  /// manifold.
  virtual bool isNormalized(const ConstVectorRef & /*x*/) const { return true; }

  /// @brief    Return an object representing the tangent space as a manifold.
  TangentSpaceType tangentSpace() const {
    return TangentSpaceType(this->ndx());
  }

  /// @name     Operations

  /// @brief Manifold integration operation \f$x \oplus v\f$
  void integrate(const ConstVectorRef &x, const ConstVectorRef &v,
                 VectorRef out) const;

  /// @brief    Jacobian of the integation operation.
  void Jintegrate(const ConstVectorRef &x, const ConstVectorRef &v,
                  MatrixRef Jout, int arg) const;

  /// @brief    Perform the parallel transport operation
  ///
  void JintegrateTransport(const ConstVectorRef &x, const ConstVectorRef &v,
                           MatrixRef Jout, int arg) const {
    assert(Jout.rows() == v.size());
    JintegrateTransport_impl(x, v, Jout, arg);
  }

  /// @brief Manifold difference/retraction operation \f$x_1 \ominus x_0\f$
  void difference(const ConstVectorRef &x0, const ConstVectorRef &x1,
                  VectorRef out) const;

  /// @brief    Jacobian of the retraction operation.
  void Jdifference(const ConstVectorRef &x0, const ConstVectorRef &x1,
                   MatrixRef Jout, int arg) const;

  void interpolate(const ConstVectorRef &x0, const ConstVectorRef &x1,
                   const Scalar &u, VectorRef out) const;

  /// \name Allocated overloads.
  /// \{

  /// @copybrief integrate()
  ///
  /// Out-of-place variant of integration operator.
  [[nodiscard]] VectorXs integrate(const ConstVectorRef &x,
                                   const ConstVectorRef &v) const {
    VectorXs out(nx());
    integrate_impl(x, v, out);
    return out;
  }

  /// @copybrief difference()
  ///
  /// Out-of-place version of diff operator.
  [[nodiscard]] VectorXs difference(const ConstVectorRef &x0,
                                    const ConstVectorRef &x1) const {
    VectorXs out(ndx());
    difference_impl(x0, x1, out);
    return out;
  }

  /// @copybrief interpolate_impl()
  [[nodiscard]] VectorXs interpolate(const ConstVectorRef &x0,
                                     const ConstVectorRef &x1,
                                     const Scalar &u) const {
    VectorXs out(nx());
    interpolate_impl(x0, x1, u, out);
    return out;
  }

  /// \}

protected:
  int nx_;
  int ndx_;

  /// Perform the manifold integration operation.
  virtual void integrate_impl(const ConstVectorRef &x, const ConstVectorRef &v,
                              VectorRef out) const = 0;

  virtual void Jintegrate_impl(const ConstVectorRef &x, const ConstVectorRef &v,
                               MatrixRef Jout, int arg) const = 0;

  virtual void JintegrateTransport_impl(const ConstVectorRef &x,
                                        const ConstVectorRef &v, MatrixRef Jout,
                                        int arg) const = 0;

  /// Implementation of the manifold retraction operation.
  virtual void difference_impl(const ConstVectorRef &x0,
                               const ConstVectorRef &x1,
                               VectorRef out) const = 0;

  virtual void Jdifference_impl(const ConstVectorRef &x0,
                                const ConstVectorRef &x1, MatrixRef Jout,
                                int arg) const = 0;

  /// @brief    Interpolation operation.
  virtual void interpolate_impl(const ConstVectorRef &x0,
                                const ConstVectorRef &x1, const Scalar &u,
                                VectorRef out) const {
    // default implementation
    integrate(x0, u * difference(x0, x1), out);
  }

  virtual void neutral_impl(VectorRef out) const {
    assert(out.size() == nx());
    out.setZero();
  }

  virtual void rand_impl(VectorRef out) const {
    assert(out.size() == nx());
    out.setRandom();
  }
};

} // namespace aligator

#include "aligator/core/manifold-base.hxx"
