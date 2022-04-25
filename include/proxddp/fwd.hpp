/// @file   Forward declarations.
#pragma once

#include "proxnlp/fwd.hpp"

/// Main package namespace.
namespace proxddp
{

// Use the standard library shared_ptr.
using std::shared_ptr;

using proxnlp::math_types;

using proxnlp::ManifoldAbstractTpl;
using proxnlp::ConstraintSetBase;


// fwd StageModelTpl
template<typename Scalar>
struct StageModelTpl;


template<typename Scalar>
struct NodeDataTpl;


/// Math utilities
namespace math
{
  using namespace proxnlp::math;
} // namespace math

} // namespace proxddp

