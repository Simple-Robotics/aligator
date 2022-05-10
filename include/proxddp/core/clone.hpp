#pragma once

#include "proxddp/fwd.hpp"


namespace proxddp
{
  template<typename T, typename concrete = void>
  struct cloneable;

  /// @brief CRTP trait which makes a class cloneable.
  ///
  /// @details  Trait CRTP struct, make T inherit from this to endow it with clone().
  ///
  /// @warning This variant requires the existence of a copy constructor for type T.
  template<typename T>
  struct cloneable<T, void>
  {
  public:
    virtual ~cloneable() = default;
    shared_ptr<T> clone() const
    {
      return shared_ptr<T>(static_cast<T*>(this->clone_impl()));
    }

  protected:
    virtual cloneable* clone_impl() const
    {
      return new T(static_cast<const T&>(*this));
    }
  };

  /// @copybrief cloneable
  /// @details This version injects the concrete type to provide
  /// a default implementation of clone_impl().
  /// @warning This requires type concrete to have a copy constructor.
  template<typename base_type, typename concrete>
  struct cloneable : base_type
  {
  public:
    virtual ~cloneable() = default;
    shared_ptr<concrete> clone() const
    {
      return shared_ptr<concrete>(static_cast<concrete*>(this->clone_impl()));
    }

  protected:
    virtual cloneable* clone_impl() const override
    {
      return new concrete(*this);
    }
  };
} // namespace proxddp

