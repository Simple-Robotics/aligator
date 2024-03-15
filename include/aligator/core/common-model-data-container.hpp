/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
/// @file common-model-data-container.hpp
/// @brief Definition of CommonModelDataContainer
#pragma once

#include "aligator/fwd.hpp"

#include "aligator/core/common-model-abstract.hpp"

#include <boost/type_index/ctti_type_index.hpp>

#include <vector>
#include <string>

namespace aligator {

// @brief Store all CommonModelData associated with
// a stage.
// Add some helper methods to update all CommonModel for new x, u.
template <typename _Scalar> class CommonModelDataContainerTpl {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Data = CommonModelDataTpl<Scalar>;
  using DataPtr = shared_ptr<Data>;

  struct value_type {
    value_type(std::string ti, std::shared_ptr<Data> m)
        : type_index(std::move(ti)), data(std::move(m)) {}

    std::string type_index;
    std::shared_ptr<Data> data;
  };

  using container_type = std::vector<value_type>;

  CommonModelDataContainerTpl() = default;
  CommonModelDataContainerTpl(container_type datas)
      : datas_(std::move(datas)) {}

  /// @return Number of contained models.
  std::size_t size() const { return datas_.size(); }

  const value_type &at(std::size_t index) const { return datas_.at(index); }

  const value_type &operator[](std::size_t index) const {
    return datas_[index];
  }

  /// @return CommonModelData pointer associated with CommonType.
  /// @throw std::runtime_error When the CommonType is not contained.
  template <typename CommonType> typename CommonType::Data *getData() const {
    std::string type_index =
        boost::typeindex::ctti_type_index::type_id<CommonType>().pretty_name();
    auto it = std::find_if(datas_.begin(), datas_.end(),
                           [&type_index](const value_type &value) {
                             return value.type_index == type_index;
                           });

    if (it == datas_.end()) {
      ALIGATOR_RUNTIME_ERROR(fmt::format(
          "{} CommonModel is not initialized or doesn't exists", type_index));
    }

    return static_cast<typename CommonType::Data *>(it->data.get());
  }

  /// @return CommonModelData pointer associated with CommonType.
  /// @throw std::runtime_error When the CommonType is not contained.
  DataPtr getDataFromTypeIndexName(const std::string &key) const {
    auto it = std::find_if(
        datas_.begin(), datas_.end(),
        [&key](const value_type &value) { return value.type_index == key; });

    if (it == datas_.end()) {
      ALIGATOR_RUNTIME_ERROR(fmt::format(
          "{} CommonModel is not initialized or doesn't exists", key));
    }

    return it->data;
  }

private:
  container_type datas_;
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/core/common-model-data-container.txx"
#endif
