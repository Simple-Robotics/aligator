#include "proxddp/context.hpp"
#include "./history-callback.hpp"

namespace proxddp {
namespace helpers {

extern template struct HistoryCallback<context::Scalar>;

} // namespace helpers
} // namespace proxddp
