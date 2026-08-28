/// @file
/// @copyright Copyright (C) 2025 INRIA
#pragma once

#include "aligator/fwd.hpp"

namespace aligator {

/// @brief Type traits for a Lie group. Usually required are the Lie group
/// dimension, corresponding Lie Algebra, etc.
template <typename Derived> struct LieGroupTraits {
  using Scalar = typename Derived::Scalar;
  static constexpr int Dim = Derived::Dim;
};

#define ALIGATOR_LIEGROUP_TRAITS_DEFINE(Traits)                                \
  using Scalar = typename Traits::Derived;                                     \
  static constexpr int Dim = Traits::Dim;                                      \
  using LieAlgebraType = typename Traits::LieAlgebraType;                      \
  using DualLieAlgebraType = typename Traits::DualLieAlgebraType

template <typename Derived> struct LieGroupBase {
  using Traits = LieGroupTraits<Derived>;
  ALIGATOR_LIEGROUP_TRAITS_DEFINE(Traits);

  Derived &derived() { return static_cast<Derived &>(*this); }

  const Derived &derived() const { return static_cast<const Derived &>(*this); }

  Derived &const_cast_derived() const {
    return const_cast<Derived &>(derived());
  }
};

} // namespace aligator
