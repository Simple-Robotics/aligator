/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
/// @file common-model-builder-container.hpp
/// @brief Definition of CommonModelBuilderContainer
#pragma once

#include "aligator/fwd.hpp"
#include "aligator/core/common-model-abstract.hpp"
#include "aligator/core/common-model-container.hpp"

#include <boost/type_index/ctti_type_index.hpp>

#include <string>
#include <unordered_map>

namespace aligator {

/// @brief Store all CommonModelBuilder associated with a stage.
template <typename _Scalar> class CommonModelBuilderContainerTpl {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Model = CommonModelTpl<Scalar>;
  using Data = CommonModelDataTpl<Scalar>;
  using Builder = CommonModelBuilderTpl<Scalar>;
  using BuilderPtr = shared_ptr<Builder>;
  using Container = CommonModelContainerTpl<Scalar>;

  using container_type =
      std::unordered_map<std::string, std::shared_ptr<Builder>>;

  /// @return CommonModelBuilderTpl pointer associated with CommonType.
  template <typename CommonType> typename CommonType::Builder &get() {
    auto key = boost::typeindex::ctti_type_index::type_id<CommonType>();
    // If insertion took place we create the new builder
    auto [it, inserted] = builders_.try_emplace(key.pretty_name(), nullptr);
    if (inserted) {
      it->second = std::make_shared<typename CommonType::Builder>();
    }
    return static_cast<typename CommonType::Builder &>(*it->second);
  }

  /// @param key Key name. User must use ctti_type_index::pretty_name to
  /// retrieve Builder added with \p get method.
  /// @param builder Builder to add if the key doesn't exists.
  /// @return CommonModelBuilderTpl shared pointer associated with a key.
  BuilderPtr getFromTypeIndexName(const std::string &key, BuilderPtr builder) {
    // If insertion took place we create the new builder
    auto [it, inserted] = builders_.try_emplace(key, nullptr);
    if (inserted) {
      it->second = builder;
    }
    return it->second;
  }

  /// Create a CommonModelContainerTpl from all configured builder.
  /// @warning This method must be called by StageModel only.
  Container createCommonModelContainer() const {
    typename Container::container_type container;
    for (const auto &b : builders_) {
      auto model = b.second->build();
      container.emplace_back(b.first, std::move(model));
    }
    return Container(std::move(container));
  }

private:
  container_type builders_;
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/core/common-model-builder-container.txx"
#endif
