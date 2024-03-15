/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
/// @file common-model-abstract.hpp
/// @brief Base definitions for common computation functions.
#pragma once

#include "aligator/fwd.hpp"

#include <memory>

namespace aligator {

/** @brief Common computation between dynamics, costs and constraints.
 */
template <typename _Scalar> struct CommonModelTpl {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Data = CommonModelDataTpl<Scalar>;
  using Builder = CommonModelBuilderTpl<Scalar>;

  /// @brief Evaluate the common model.
  virtual void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                        Data &data) const = 0;

  /// @brief Compute the common model gradients.
  virtual void computeGradients(const ConstVectorRef &x,
                                const ConstVectorRef &u, Data &data) const = 0;

  /// @brief Compute the common model Hessians.
  // TODO useful ?
  virtual void computeHessians(const ConstVectorRef &x, const ConstVectorRef &u,
                               Data &data) const = 0;

  virtual std::shared_ptr<Data> createData() const {
    return std::make_shared<Data>();
  }

  virtual ~CommonModelTpl() = default;
};

/// @brief Data structure for CommonModelTpl
template <typename _Scalar> struct CommonModelDataTpl {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  virtual ~CommonModelDataTpl() = default;
};

/// @brief Builder/Factory for CommonModelTpl
template <typename _Scalar> class CommonModelBuilderTpl {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Model = CommonModelTpl<Scalar>;

  virtual std::shared_ptr<Model> build() const = 0;

  virtual ~CommonModelBuilderTpl() = default;
};

} // namespace aligator

// TODO template instantiating
