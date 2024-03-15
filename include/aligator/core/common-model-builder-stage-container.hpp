/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
/// @file common-model-builder-stage-container.hpp
/// @brief Definition of CommonModelBuilderStageContainer
#pragma once

#include "aligator/fwd.hpp"
#include "aligator/core/common-model-abstract.hpp"
#include "aligator/core/common-model-stage-container.hpp"

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

/** @brief Store all CommonModelBuilder associated with a stage.
 */
template <typename _Scalar> class CommonModelBuilderStageContainerTpl {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Model = CommonModelTpl<Scalar>;
  using Data = CommonModelDataTpl<Scalar>;
  using Builder = CommonModelBuilderTpl<Scalar>;
  using Container = CommonModelStageContainerTpl<Scalar>;

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

  /// Create a CommonModelStageContainerTpl from all configured builder
  Container create_common_container() const {
    typename Container::container_type container;
    for (const auto &b : builders_) {
      auto model = b.second->build();
      auto data = model->createData();
      container.emplace_back(b.first, std::move(model), std::move(data));
    }
    return Container(std::move(container));
  }

private:
  container_type builders_;
};

/** @brief This class will be used by dynamics/costs/constraints to
 * create/retrieve builder without access to other
 * CommonModelBuilderStageContainerTpl methods.
 */
template <typename _Scalar> class CommonModelBuilderStageContainerHandleTpl {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Container = CommonModelBuilderStageContainerTpl<Scalar>;

  CommonModelBuilderStageContainerHandleTpl(Container &container)
      : container_(container) {}

  /// \return CommonModelBuilder pointer associated with CommonType
  template <typename CommonType> typename CommonType::Builder *get() {
    return container_.template get<CommonType>();
  }

private:
  Container &container_;
};

} // namespace aligator

// TODO template instantiation
