/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/fwd.hpp"
#include "aligator/core/function-abstract.hpp"
#include "aligator/core/cost-abstract.hpp"
#include "aligator/modelling/state-error.hpp"

#include <fmt/ostream.h>

namespace aligator {

/// Data struct for composite costs.
template <typename Scalar>
struct CompositeCostDataTpl : CostDataAbstractTpl<Scalar> {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = CostDataAbstractTpl<Scalar>;
  using StageFunctionData = StageFunctionDataTpl<Scalar>;
  using RowMatrixXs = Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>;

  shared_ptr<StageFunctionData> residual_data;
  RowMatrixXs JtW_buf;
  VectorXs Wv_buf;
  CompositeCostDataTpl(const int ndx, const int nu,
                       shared_ptr<StageFunctionData> rdata)
      : Base(ndx, nu), residual_data(rdata), JtW_buf(ndx + nu, rdata->nr),
        Wv_buf(rdata->nr) {
    JtW_buf.setZero();
    Wv_buf.setZero();
  }
};

} // namespace aligator

#include "./quad-residual-cost.hpp"
#include "./log-residual-cost.hpp"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./composite-costs.txx"
#endif
