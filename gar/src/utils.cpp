/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#include "aligator/gar/utils.hpp"

namespace aligator {
namespace gar {
template auto
lqrDenseMatrix<context::Scalar>(const LQRProblemTpl<context::Scalar> &,
                                context::Scalar, context::Scalar);
} // namespace gar
} // namespace aligator
