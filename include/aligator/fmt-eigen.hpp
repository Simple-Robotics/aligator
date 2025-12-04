/// @file
/// @copyright Copyright (C) 2025 LAAS-CNRS, INRIA
#pragma once

#include "math.hpp"
#include <fmt/ostream.h>
#include <fmt/ranges.h>

/// Specialize fmt::formatter using the operator<< implementation for Eigen
/// types.
template <typename MatrixType>
struct fmt::formatter<MatrixType,
                      ::aligator::enable_if_eigen_dense_t<MatrixType, char>>
    : fmt::formatter<std::ostream, char> {};

template <typename MatrixType>
struct fmt::is_range<MatrixType,
                     ::aligator::enable_if_eigen_dense_t<MatrixType, char>>
    : std::false_type {};

template <typename MatrixType>
struct fmt::formatter<Eigen::WithFormat<MatrixType>>
    : public fmt::formatter<std::ostream, char> {};
