/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"

#include "aligator/core/common-model-abstract.hpp"
#include "aligator/core/common-model-container.hpp"
#include "aligator/core/common-model-builder-container.hpp"
#include "aligator/core/common-model-data-container.hpp"
#include "aligator/python/common-model.hpp"

namespace aligator {
namespace python {

namespace {

void exposeCommonModelAbstract() {
  using context::CommonModel;
  using context::CommonModelBuilder;
  using context::CommonModelData;
  using context::Scalar;

  bp::register_ptr_to_python<shared_ptr<CommonModel>>();
  bp::class_<internal::PyCommonModel<>, boost::noncopyable>(
      "CommonModel",
      "Common computation between dynamics, costs and constraints.")
      .def("evaluate", bp::pure_virtual(&CommonModel::evaluate),
           bp::args("self", "x", "u", "data"), "Evaluate the cost function.")
      .def("computeGradients", bp::pure_virtual(&CommonModel::computeGradients),
           bp::args("self", "x", "u", "data"),
           "Compute the cost function gradients.")
      .def("computeHessians", bp::pure_virtual(&CommonModel::computeHessians),
           bp::args("self", "x", "u", "data"),
           "Compute the cost function hessians.")
      .def(CreateDataPythonVisitor<CommonModel>());

  bp::register_ptr_to_python<shared_ptr<CommonModelData>>();
  bp::class_<internal::PyCommonModelData<>, boost::noncopyable>(
      "CommonModelData", "Data structure for CommonModel.");

  bp::register_ptr_to_python<shared_ptr<CommonModelBuilder>>();
  bp::class_<internal::PyCommonModelBuilder<>, boost::noncopyable>(
      "CommonModelBuilder", "Builder/Factory for CommonModel.")
      .def("build", bp::pure_virtual(&CommonModelBuilder::build),
           bp::args("self"), "Create a CommonModel instance.");
}

void exposeCommonModelBuilderContainer() {
  using context::CommonModelBuilderContainer;

  /// TODO Find a way to manage Python CommonModelBuilder in get method
  /// TODO Allow to call get method in Python
  bp::class_<CommonModelBuilderContainer>(
      "CommonModelBuilderContainer",
      "Store all CommonModelBuilder associated with a stage.")
      .def("createCommonModelContainer",
           &CommonModelBuilderContainer::createCommonModelContainer,
           bp::args("self"),
           "Create a CommonModelContainerTpl from all configured builder.");
}

void exposeCommonModelDataContainer() {
  using context::CommonModelDataContainer;

  /// TODO Allow to call getData method in Python
  bp::class_<CommonModelDataContainer>(
      "CommonModelDataContainer",
      "Store all CommonModelData associated with a stage.")
      .def("size", &CommonModelDataContainer::size, bp::args("self"),
           "Number of CommonModel inside the container.")
      .def("__getitem__", &CommonModelDataContainer::at,
           bp::return_internal_reference<>(), bp::args("self", "index"),
           "Get item index.");
}

void exposeCommonModelContainer() {
  using context::CommonModelContainer;

  bp::class_<CommonModelContainer>(
      "CommonModelContainer", "Store all CommonModel associated with a stage.")
      .def("size", &CommonModelContainer::size, bp::args("self"),
           "Number of CommonModel inside the container.")
      .def("__getitem__", &CommonModelContainer::at,
           bp::return_internal_reference<>(), bp::args("self", "index"),
           "Get item index.")
      .def(CreateDataPythonVisitor<CommonModelContainer>());
}

} // namespace

void exposeCommonModel() {
  exposeCommonModelAbstract();
  exposeCommonModelBuilderContainer();
  exposeCommonModelDataContainer();
  exposeCommonModelContainer();
}

} // namespace python
} // namespace aligator
