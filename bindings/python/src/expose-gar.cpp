/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"
#include "aligator/gar/riccati.hpp"

#include "aligator/python/utils.hpp"

namespace aligator {
namespace python {
using riccati_bwd_t = gar::ProximalRiccatiSolverBackward<context::Scalar>;
using riccati_fwd_t = gar::ProximalRiccatiSolverForward<context::Scalar>;
using knot_t = LQRKnot<context::Scalar>;
using stage_solve_data_t = riccati_bwd_t::stage_solve_data_t;
} // namespace python
} // namespace aligator

namespace eigenpy {
namespace internal {
template <>
struct has_operator_equal<::aligator::python::knot_t> : boost::false_type {};
template <>
struct has_operator_equal<::aligator::python::stage_solve_data_t>
    : boost::false_type {};
} // namespace internal
} // namespace eigenpy

namespace aligator {
namespace python {

using knot_vec_t = std::vector<knot_t>;

void exposeGAR() { bp::scope ns = get_namespace("gar"); }

} // namespace python
} // namespace aligator
