#pragma once

#include "proxddp/fwd.hpp"

namespace proxddp {
/// @brief Mixin which makes a class cloneable.
///
/// @tparam T         The type to make cloneable, or its parent through which
/// clone() is called.
/// @tparam concrete  The concrete type to clone, or nothing if T is a top node
/// in the class hierarchy.
template <typename T, typename concrete = void> struct cloneable;

/// @copybrief cloneable
/// @tparam T The type to make clonable.
/// @warning  This variant requires the existence of a <a
/// href="https://en.cppreference.com/w/cpp/language/copy_constructor">copy
/// constructor</a> for type T.
///           Most of the time, the default copy constructor shall suffice if it
///           is not deleted.
template <typename T> struct cloneable<T, void> {
public:
  virtual ~cloneable() = default;
  shared_ptr<T> clone() const {
    return shared_ptr<T>(static_cast<T *>(this->clone_impl()));
  }

protected:
  virtual cloneable *clone_impl() const {
    return new T(static_cast<const T &>(*this));
  }
};

/// @warning  This requires type @c concrete to have a <a
/// href="https://en.cppreference.com/w/cpp/language/copy_constructor">copy
/// constructor</a>.
///           Often, the compiler-generated default copy constructor shall
///           suffice (if it is not deleted).
template <typename T, typename concrete> struct cloneable : T {
public:
  virtual ~cloneable() = default;
  shared_ptr<concrete> clone() const {
    return shared_ptr<concrete>(static_cast<concrete *>(this->clone_impl()));
  }

protected:
  virtual cloneable *clone_impl() const override { return new concrete(*this); }
};
} // namespace proxddp
