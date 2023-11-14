/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/gar/riccati.hpp"
#include "aligator/gar/helpers.hpp"

ALIGATOR_DYNAMIC_TYPEDEFS(double);
using vecvec_t = std::vector<VectorXs>;
using prox_riccati_t = aligator::gar::ProximalRiccatiSolver<double>;
using problem_t = aligator::gar::LQRProblemTpl<double>;
using knot_t = aligator::gar::LQRKnotTpl<double>;
using aligator::math::infty_norm;

struct KktError {
  // dynamics error
  double dyn;
  // constraint error
  double cstr;
  double dual;
  double max;
};

KktError
compute_kkt_error(const problem_t &problem, const vecvec_t &xs,
                  const vecvec_t &us, const vecvec_t &vs, const vecvec_t &lbdas,
                  const boost::optional<ConstVectorRef> &theta_ = boost::none);

auto wishart_dist_matrix(uint n, uint p);

problem_t generate_problem(const ConstVectorRef &x0, uint horz, uint nx,
                           uint nu);
