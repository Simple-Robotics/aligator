/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/fwd.hpp"

namespace aligator {
/// @brief Mixin which makes a class/class hierarchy cloneable.
/// @details Inheriting from Cloneable<T> creates a function clone() returning
/// shared_ptr<T>. Child classes of T only need to implement the virtual member
/// function clone_impl() which returns the covariant pointer type T*.
/// @tparam The class (or base class in a hierarchy) we want to make cloneable.
template <typename T> struct Cloneable {
  // non-virtual
  shared_ptr<T> clone() const {
    return shared_ptr<T>(static_cast<T *>(clone_impl()));
  }

protected:
  virtual Cloneable *clone_impl() const = 0;
};

} // namespace aligator
