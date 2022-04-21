/// @file   Forward declarations.
#pragma once

#include "proxnlp/fwd.hpp"

/// Main package namespace.
namespace proxddp
{

// Use the standard library shared_ptr.
using std::shared_ptr;

using proxnlp::ManifoldAbstractTpl;
using proxnlp::CostFunctionBaseTpl;
using proxnlp::C2FunctionTpl;
using proxnlp::ConstraintSetBase;


// fwd NodeTpl
template<typename Scalar>
struct NodeTpl;

/// Math utilities
namespace math
{
  using namespace proxnlp::math;
} // namespace math

} // namespace proxddp

