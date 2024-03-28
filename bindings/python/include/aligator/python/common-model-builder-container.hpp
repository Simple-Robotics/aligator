/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/python/fwd.hpp"
#include "aligator/core/common-model-builder-container.hpp"

namespace aligator {
namespace python {
namespace internal {

class PyCommonModelBuilderContainerWrapper {
public:
  using Scalar = context::Scalar;
  using BuilderContainer = CommonModelBuilderContainerTpl<Scalar>;
  using Builder = CommonModelBuilderTpl<Scalar>;
  using BuilderPtr = shared_ptr<CommonModelBuilderTpl<Scalar>>;
  using Container = CommonModelContainerTpl<Scalar>;

  PyCommonModelBuilderContainerWrapper(BuilderContainer &c) : container_(c) {}

  BuilderPtr get(bp::object object) {
    BuilderPtr builder = bp::extract<BuilderPtr>(object);
    std::string type_index =
        boost::python::extract<std::string>(object.attr("__name__"));
    return container_.getFromTypeIndexRawName(type_index, builder);
  }

  Container createCommonModelContainer() const {
    return container_.createCommonModelContainer();
  }

private:
  BuilderContainer &container_;
};

} // namespace internal
} // namespace python
} // namespace aligator
