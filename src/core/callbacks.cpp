#include "proxddp/helpers/history-callback.hpp"
#include "proxddp/helpers/linesearch-callback.hpp"

namespace proxddp {
namespace helpers {

template struct HistoryCallback<context::Scalar>;
template struct LinesearchCallback<context::Scalar>;

} // namespace helpers
} // namespace proxddp
