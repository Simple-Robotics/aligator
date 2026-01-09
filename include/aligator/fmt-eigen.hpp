/// @file
/// @copyright Copyright (C) 2025 LAAS-CNRS, INRIA
#pragma once

#include "math.hpp"

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

/// Specialize fmt::formatter using the operator<< implementation for Eigen
/// types.

// ostream_formatter has been introduced in FMT 9
#if FMT_VERSION >= 90000
template <typename MatrixType>
struct fmt::formatter<MatrixType,
                      ::aligator::enable_if_eigen_dense_t<MatrixType, char>>
    : fmt::ostream_formatter {};

template <typename MatrixType>
struct fmt::is_range<MatrixType,
                     ::aligator::enable_if_eigen_dense_t<MatrixType, char>>
    : std::false_type {};

template <typename MatrixType>
struct fmt::formatter<Eigen::WithFormat<MatrixType>>
    : public fmt::ostream_formatter {};
#else
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
#endif // if FMT_VERSION >= 90000
