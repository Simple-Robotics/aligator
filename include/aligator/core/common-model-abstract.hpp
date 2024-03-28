/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
/// @file common-model-abstract.hpp
/// @brief Base definitions for common computation functions.
#pragma once

#include "aligator/fwd.hpp"

#include <memory>

namespace aligator {

/// @brief Common computation between dynamics, costs and constraints.
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
  using Base = CommonModelDataTpl<Scalar>;

  virtual ~CommonModelDataTpl() = default;
};

/// @brief Builder/Factory for CommonModelTpl
template <typename _Scalar> class CommonModelBuilderTpl {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Model = CommonModelTpl<Scalar>;
  using Base = CommonModelBuilderTpl<Scalar>;

  virtual std::shared_ptr<Model> build() const = 0;

  virtual ~CommonModelBuilderTpl() = default;
};

} // namespace aligator
//
#define ALIGATOR_COMMON_MODEL_TYPEDEFS(Scalar, _Base, _Data, _Builder)         \
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);                                           \
  using Base = _Base<Scalar>;                                                  \
  using Data = _Data<Scalar>;                                                  \
  using Builder = _Builder<Scalar>;                                            \
  using BaseData = typename Base::Data

#define ALIGATOR_COMMON_DATA_TYPEDEFS(Scalar, _Model)                          \
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);                                           \
  using Model = _Model<Scalar>;                                                \
  using Base = typename Model::Data::Base

#define ALIGATOR_COMMON_BUILDER_TYPEDEFS(Scalar, _Model)                       \
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);                                           \
  using Model = _Model<Scalar>;                                                \
  using BaseModel = typename Model::Base;                                      \
  using Base = typename Model::Builder::Base;                                  \
  using Self = typename Model::Builder

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/core/common-model-abstract.txx"
#endif
