#pragma once

#include "proxddp/context.hpp"

namespace proxddp {

extern template PDALFunction<context::Scalar>::PDALFunction(context::SolverProxDDP const *);

}
