/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
/// @file common-model-stage-container.hpp
/// @brief Definition of CommonModelContainer
#pragma once

#include "aligator/fwd.hpp"

#include "aligator/core/common-model-abstract.hpp"

#include <boost/type_index/ctti_type_index.hpp>

#include <vector>

namespace aligator {

// @brief Store all CommonModel and associated CommonModelData associated with
// a stage.
// Add some helper methods to update all CommonModel for new x, u.
template <typename _Scalar> class CommonModelContainerTpl {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Model = CommonModelTpl<Scalar>;
  using Data = CommonModelDataTpl<Scalar>;

  struct Value {
    Value(boost::typeindex::ctti_type_index ti, std::shared_ptr<Model> m,
          std::shared_ptr<Data> d)
        : type_index(ti), model(std::move(m)), data(std::move(d)) {}

    boost::typeindex::ctti_type_index type_index;
    std::shared_ptr<Model> model;
    std::shared_ptr<Data> data;
  };

  using container_type = std::vector<Value>;

  CommonModelContainerTpl(container_type models) : models_(std::move(models)) {}

  /// @return Number of contained models
  std::size_t size() const { return models_.size(); }

  const Value &at(std::size_t index) const { return models_.at(index); }

  /// Call CommonModelTpl::evaluate for each stored model
  void evaluateAll(const ConstVectorRef &x, const ConstVectorRef &u) {
    for (auto &v : models_) {
      v.model->evaluate(x, u, *v.data);
    }
  }

  /// Call CommonModelTpl::computeGradients for each stored model
  void computeAllGradients(const ConstVectorRef &x, const ConstVectorRef &u) {
    for (auto &v : models_) {
      v.model->computeGradients(x, u, *v.data);
    }
  }

  /// Call CommonModelTpl::computeHessians for each stored model
  void computeAllHessians(const ConstVectorRef &x, const ConstVectorRef &u) {
    for (auto &v : models_) {
      v.model->computeHessians(x, u, *v.data);
    }
  }

  /// @return CommonModelData pointer associated with CommonType
  /// @throw std::runtime_error When the CommonType is not contained
  template <typename CommonType>
  typename CommonType::Data *get_common_data() const {
    auto type_index = boost::typeindex::ctti_type_index::type_id<CommonType>();
    auto it = std::find_if(models_.begin(), models_.end(),
                           [&type_index](const Value &value) {
                             return value.type_index.equal(type_index);
                           });

    if (it == models_.end()) {
      ALIGATOR_RUNTIME_ERROR(
          fmt::format("{} CommonModel is not initialized or doesn't exists"));
    }

    return static_cast<typename CommonType::Data *>(it->data.get());
  }

private:
  container_type models_;
};

} // namespace aligator

// TODO template instantiation
