#include "proxddp/python/fwd.hpp"

#include "proxddp/solver-proxddp.hpp"


namespace proxddp
{
  namespace python
  {

    void exposeSolvers()
    {
      using context::Scalar;
      using context::ShootingProblem;
      using Workspace = WorkspaceTpl<Scalar>;
      using Results = ResultsTpl<Scalar>;

      bp::class_<Workspace>(
        "Workspace", "Workspace for ProxDDP.",
        bp::init<const ShootingProblem&>(bp::args("self", "problem"))
        )
        .def_readonly("value_params", &Workspace::value_params)
        .def_readonly("q_params", &Workspace::q_params)
        .def_readonly("kkt_matrix_buffer_", &Workspace::kktMatrixFull_)
        .def_readonly("gains", &Workspace::gains_)
        ;

      bp::class_<Results>(
        "Results", "Results struct for proxDDP.", bp::init<const ShootingProblem&>()
      )
        .def_readonly("xs", &Results::xs_)
        .def_readonly("us", &Results::us_)
        .def_readonly("lams", &Results::lams_)
        .def_readonly("co_state", &Results::co_state_)
      ;

      bp::def("forward_pass", &forward_pass<Scalar>, "Perform the forward pass");
      bp::def("try_step", &try_step<Scalar>, "Try a step of size \\alpha.");

    }
    
  } // namespace python
} // namespace proxddp

