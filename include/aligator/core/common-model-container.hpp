/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
/// @file common-model-container.hpp
/// @brief Definition of CommonModelContainer
#pragma once

#include "aligator/fwd.hpp"

#include "aligator/core/common-model-abstract.hpp"
#include "aligator/core/common-model-data-container.hpp"

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
  using CommonModelDataContainer = CommonModelDataContainerTpl<Scalar>;

  struct value_type {
    value_type(boost::typeindex::ctti_type_index ti, std::shared_ptr<Model> m)
        : type_index(ti), model(std::move(m)) {}

    boost::typeindex::ctti_type_index type_index;
    std::shared_ptr<Model> model;
  };

  using container_type = std::vector<value_type>;

  CommonModelContainerTpl(container_type models) : models_(std::move(models)) {}

  /// @return Number of contained models.
  std::size_t size() const { return models_.size(); }

  const value_type &at(std::size_t index) const { return models_.at(index); }

  const value_type &operator[](std::size_t index) const {
    return models_[index];
  }

  CommonModelDataContainer createData() const {
    typename CommonModelDataContainer::container_type container;
    container.reserve(models_.size());
    for (const auto &v : models_) {
      container.emplace_back(v.type_index, v.model->createData());
    }
    return CommonModelDataContainer(std::move(container));
  }

private:
  container_type models_;
};

} // namespace aligator

// TODO template instantiation
