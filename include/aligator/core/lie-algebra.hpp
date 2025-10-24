/// @file
/// @copyright Copyright (C) 2025 INRIA
#pragma once

#include "aligator/fwd.hpp"

namespace aligator {

template <typename Derived> struct LieAlgebraTraits;

/// @brief Base CRTP for Lie algebras.
template <typename Derived> struct LieAlgebraBase {
  using Traits = LieAlgebraTraits<Derived>;
};

} // namespace aligator
