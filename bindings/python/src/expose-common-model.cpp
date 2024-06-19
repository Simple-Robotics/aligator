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
      .def("createData", &CommonModel::createData,
           &internal::PyCommonModel<>::default_createData, bp::args("self"),
           "Create a data object.");

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

  bp::class_<CommonModelBuilderContainer>(
      "CommonModelBuilderContainer",
      "Store all CommonModelBuilder associated with a stage.")
      .def("createCommonModelContainer",
           &CommonModelBuilderContainer::createCommonModelContainer,
           bp::args("self"),
           "Create a CommonModelContainerTpl from all configured builder.")
      .def("get", &CommonModelBuilderContainer::getFromTypeIndexName,
           bp::args("self", "key", "builder"),
           "Get a CommonModelBuilder from a key or the provided buider if the "
           "key doesn't exists.");
}

void exposeCommonModelDataContainer() {
  using context::CommonModelDataContainer;

  bp::class_<CommonModelDataContainer>(
      "CommonModelDataContainer",
      "Store all CommonModelData associated with a stage.")
      .def("size", &CommonModelDataContainer::size, bp::args("self"),
           "Number of CommonModel inside the container.")
      .def("__getitem__", &CommonModelDataContainer::at,
           bp::return_internal_reference<>(), bp::args("self", "index"),
           "Get item index.")
      .def("get", &CommonModelDataContainer::getDataFromTypeIndexName,
           bp::args("self", "key"), "Get a CommonModelData from a key.");
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
      .def("createData", &CommonModelContainer::createData, bp::args("self"),
           "Create a data object.")
      .def("evaluate", &CommonModelContainer::evaluate,
           bp::args("self", "x", "u", "data"),
           "Call evaluate for each contained models.")
      .def("computeGradients", &CommonModelContainer::computeGradients,
           bp::args("self", "x", "u", "data"),
           "Call computeGradients for each contained models.")
      .def("computeHessians", &CommonModelContainer::computeHessians,
           bp::args("self", "x", "u", "data"),
           "Call computeHessians for each contained models.");
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
