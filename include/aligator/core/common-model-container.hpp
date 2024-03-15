/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
/// @file common-model-container.hpp
/// @brief Definition of CommonModelContainer
#pragma once

#include "aligator/fwd.hpp"

#include "aligator/core/common-model-abstract.hpp"
#include "aligator/core/common-model-data-container.hpp"

#include <vector>
#include <string>

namespace aligator {

// @brief Store all CommonModel associated with a stage.
template <typename _Scalar> class CommonModelContainerTpl {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Model = CommonModelTpl<Scalar>;
  using Data = CommonModelDataTpl<Scalar>;
  using CommonModelDataContainer = CommonModelDataContainerTpl<Scalar>;

  struct value_type {
    value_type(std::string ti, std::shared_ptr<Model> m)
        : type_index(std::move(ti)), model(std::move(m)) {}

    std::string type_index;
    std::shared_ptr<Model> model;
  };

  using container_type = std::vector<value_type>;

  CommonModelContainerTpl() = default;
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

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                CommonModelDataContainer &datas) const {
    //  TODO assert ?
    for (std::size_t i = 0; i < models_.size(); i++) {
      models_[i].model->evaluate(x, u, *datas[i].data);
    }
  }

  void computeGradients(const ConstVectorRef &x, const ConstVectorRef &u,
                        CommonModelDataContainer &datas) const {
    //  TODO assert ?
    for (std::size_t i = 0; i < models_.size(); i++) {
      models_[i].model->computeGradients(x, u, *datas[i].data);
    }
  }

  void computeHessians(const ConstVectorRef &x, const ConstVectorRef &u,
                       CommonModelDataContainer &datas) const {
    //  TODO assert ?
    for (std::size_t i = 0; i < models_.size(); i++) {
      models_[i].model->computeHessians(x, u, *datas[i].data);
    }
  }

private:
  container_type models_;
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/core/common-model-container.txx"
#endif
