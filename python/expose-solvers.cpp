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

      bp::class_<Workspace>(
        "Workspace", "Workspace for ProxDDP.",
        bp::init<const ShootingProblem&>(bp::args("self", "problem"))
        )
        .def_readonly("value_params", &Workspace::value_params)
        .def_readonly("q_params", &Workspace::q_params)
        .def_readonly("kkt_matrix_buffer_", &Workspace::kktMatrixFull_)
        .def_readonly("gains", &Workspace::gains_)
        ;

    }
    
  } // namespace python
} // namespace proxddp

