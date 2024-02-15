#include "aligator/modelling/multibody/kinodynamics-wrapper.hpp"

namespace aligator {

template struct KinodynamicsWrapperResidualTpl<context::Scalar>;
template struct KinodynamicsWrapperDataTpl<context::Scalar>;

} // namespace aligator
