/// @file
/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#include "aligator/python/polymorphic-convertible.hpp"
#include "aligator/core/stage-model.hpp"
#include "aligator/core/stage-data.hpp"
#include "aligator/core/cost-abstract.hpp"

namespace aligator::python {

void exposeStageData() {
  using context::StageData;
  using context::StageModel;

  bp::register_ptr_to_python<shared_ptr<StageData>>();
  StdVectorPythonVisitor<std::vector<shared_ptr<StageData>>, true>::expose(
      "StdVec_StageData");

  bp::class_<StageData>("StageData", "Data struct for StageModel objects.",
                        bp::init<const StageModel &>())
      .def_readonly("cost_data", &StageData::cost_data)
      .def_readwrite("dynamics_data", &StageData::dynamics_data)
      .def_readwrite("constraint_data", &StageData::constraint_data);
}

} // namespace aligator::python
