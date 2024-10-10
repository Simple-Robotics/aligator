#include "aligator/core/traj-opt-data.hxx" // impl file

namespace aligator {

template struct TrajOptDataTpl<context::Scalar>;
template context::Scalar computeTrajectoryCost<context::Scalar>(
    const context::TrajOptData &problem_data);

} // namespace aligator
