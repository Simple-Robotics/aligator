/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/python/fwd.hpp"
#include "aligator/core/common-model-abstract.hpp"

namespace aligator {
namespace python {
namespace internal {

/// @brief Wrapper for the CommonModelTpl class and its children.
template <typename T = CommonModelTpl<context::Scalar>>
struct PyCommonModel : T, bp::wrapper<T> {
  using Scalar = context::Scalar;
  using bp::wrapper<T>::get_override;
  using Data = CommonModelDataTpl<Scalar>;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                Data &data) const override {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "evaluate", x, u, boost::ref(data));
  }

  void computeGradients(const ConstVectorRef &x, const ConstVectorRef &u,
                        Data &data) const override {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "computeGradients", x, u,
                                  boost::ref(data));
  }

  void computeHessians(const ConstVectorRef &x, const ConstVectorRef &u,
                       Data &data) const override {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "computeHessians", x, u,
                                  boost::ref(data));
  }

  shared_ptr<Data> createData() const override {
    ALIGATOR_PYTHON_OVERRIDE(shared_ptr<Data>, T, createData, );
  }

  shared_ptr<Data> default_createData() const { return T::createData(); }
};

template <typename T = CommonModelDataTpl<context::Scalar>>
struct PyCommonModelData : T, bp::wrapper<T> {};

template <typename T = CommonModelBuilderTpl<context::Scalar>>
struct PyCommonModelBuilder : T, bp::wrapper<T> {
  using Scalar = context::Scalar;
  using bp::wrapper<T>::get_override;
  using CommonModel = CommonModelTpl<Scalar>;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  std::shared_ptr<CommonModel> build() const override {
    ALIGATOR_PYTHON_OVERRIDE_PURE(shared_ptr<CommonModel>, "build", );
  }
};

} // namespace internal
} // namespace python
} // namespace aligator
