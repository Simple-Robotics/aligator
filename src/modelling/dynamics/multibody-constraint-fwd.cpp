#include "proxddp/fwd.hpp"

#ifdef PROXDDP_PINOCCHIO_V3
#include "proxddp/modelling/dynamics/multibody-constraint-fwd.hpp"

namespace proxddp {
namespace dynamics {

template struct MultibodyConstraintFwdDynamicsTpl<context::Scalar>;
template struct MultibodyConstraintFwdDataTpl<context::Scalar>;

} // namespace dynamics
} // namespace proxddp
#endif
