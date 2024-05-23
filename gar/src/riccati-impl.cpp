#include "aligator/gar/riccati-impl.hpp"

namespace aligator {
namespace gar {

template struct StageFactor<context::Scalar>;
template struct ProximalRiccatiKernel<context::Scalar>;

} // namespace gar
} // namespace aligator
