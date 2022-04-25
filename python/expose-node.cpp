#include "proxddp/python/fwd.hpp"

#include "proxddp/node.hpp"


namespace proxddp
{
namespace python
{
  void exposeNode()
  {
    using context::Scalar;
    using context::Manifold;

    bp::class_<StageModelTpl<Scalar>, boost::noncopyable>(
      "Node", "Control problem node.",
      bp::init<const shared_ptr<Manifold>&>(bp::args("self", "space")));
  }
  
} // namespace python
} // namespace proxddp

