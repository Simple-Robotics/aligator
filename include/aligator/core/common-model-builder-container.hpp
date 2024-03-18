/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
/// @file common-model-builder-container.hpp
/// @brief Definition of CommonModelBuilderContainer
#pragma once

#include "aligator/fwd.hpp"
#include "aligator/core/common-model-abstract.hpp"
#include "aligator/core/common-model-container.hpp"

#include <boost/type_index/ctti_type_index.hpp>

#include <unordered_map>

namespace aligator {

namespace internal {

/// boost::typeindex::ctti_type_index hash operator
struct HashCTTITypeIndex {
  std::size_t
  operator()(const boost::typeindex::ctti_type_index &s) const noexcept {
    return s.hash_code();
  }
};

/// boost::typeindex::ctti_type_index equality operator
struct EqualCTTITypeIndex {
  bool operator()(const boost::typeindex::ctti_type_index &s1,
                  const boost::typeindex::ctti_type_index &s2) const noexcept {
    return s1.equal(s2);
  }
};

} // namespace internal

/// @brief Store all CommonModelBuilder associated with a stage.
template <typename _Scalar> class CommonModelBuilderContainerTpl {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Model = CommonModelTpl<Scalar>;
  using Data = CommonModelDataTpl<Scalar>;
  using Builder = CommonModelBuilderTpl<Scalar>;
  using Container = CommonModelContainerTpl<Scalar>;

  using container_type =
      std::unordered_map<boost::typeindex::ctti_type_index,
                         std::shared_ptr<Builder>, internal::HashCTTITypeIndex,
                         internal::EqualCTTITypeIndex>;

  /// Return a CommonModelBuilderTpl pointer
  template <typename CommonType> typename CommonType::Builder *get() {
    auto key = boost::typeindex::ctti_type_index::type_id<CommonType>();
    // If insertion took place we create the new builder
    auto [it, inserted] = builders_.try_emplace(key, nullptr);
    if (inserted) {
      it->second = std::make_shared<typename CommonType::Builder>();
    }
    return static_cast<typename CommonType::Builder *>(it->second.get());
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

// TODO template instantiation
