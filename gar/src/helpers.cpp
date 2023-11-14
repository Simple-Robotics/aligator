/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#include "proxddp/gar/helpers.hpp"

namespace proxddp {
namespace gar {
template auto
lqrDenseMatrix<context::Scalar>(const LQRProblemTpl<context::Scalar> &,
                                context::Scalar, context::Scalar);
} // namespace gar
} // namespace proxddp
