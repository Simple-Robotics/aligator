#include "aligator/gar/riccati-impl.hpp"

namespace aligator {
namespace gar {

template struct StageFactor<context::Scalar>;
template struct ProximalRiccatiImpl<context::Scalar>;

} // namespace gar
} // namespace aligator
