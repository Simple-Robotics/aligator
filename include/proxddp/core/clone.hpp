#pragma once

#include "proxddp/fwd.hpp"

namespace proxddp {
/// @brief Mixin which makes a class cloneable.
///
/// @tparam T The type to make cloneable, or its parent through which
/// clone() is called.
/// @tparam U The concrete type to clone, or nothing if T is a top node
/// in the class hierarchy.
/// @warning  This requires type @c U to have a copy constructor.
///           The compiler-generated default one will suffice (if it is not
///           deleted).
template <typename T, typename U = void> struct Cloneable {
public:
  virtual ~Cloneable() = default;
  shared_ptr<U> clone() const {
    return shared_ptr<U>(static_cast<U *>(this->clone_impl()));
  }

protected:
  virtual U *clone_impl() const { return new U(*this); }
};

/// @copybrief cloneable
/// @tparam T The type to make clonable.
/// @warning  This variant requires the existence of a copy constructor for type
/// T.
///           Most of the time, the default copy constructor will suffice if it
///           is not deleted.
template <typename T> struct Cloneable<T, void> {
public:
  virtual ~Cloneable() = default;
  shared_ptr<T> clone() const {
    return shared_ptr<T>(static_cast<T *>(this->clone_impl()));
  }

protected:
  virtual T *clone_impl() const { return new T(static_cast<const T &>(*this)); }
};

} // namespace proxddp
