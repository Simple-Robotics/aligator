#pragma once

#include "aligator/core/manifold-base.hpp"
#include "aligator/third-party/polymorphic_cxx14.h"

#include <type_traits>

namespace aligator {

/** @brief    The cartesian product of two or more manifolds.
 */
template <typename _Scalar>
struct CartesianProductTpl : ManifoldAbstractTpl<_Scalar> {
public:
  using Scalar = _Scalar;

  using Base = ManifoldAbstractTpl<Scalar>;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base::ndx;
  using Base::nx;

  const Base &getComponent(std::size_t i) const { return *m_components[i]; }

  inline std::size_t numComponents() const { return m_components.size(); }

  template <class Concrete> inline void addComponent(const Concrete &c) {
    static_assert(
        std::is_base_of_v<Base, Concrete> ||
            std::is_same_v<Concrete, xyz::polymorphic<Base>>,
        "Input type should either be derived from ManifoldAbstractTpl or be "
        "polymorphic<ManifoldAbstractTpl>.");
    this->nx_ += _get_nx(c);
    this->ndx_ += _get_ndx(c);
    m_components.emplace_back(c);
  }

  template <class Concrete> inline void addComponent(Concrete &&c) {
    static_assert(
        std::is_base_of_v<Base, Concrete> ||
            std::is_same_v<Concrete, xyz::polymorphic<Base>>,
        "Input type should either be derived from ManifoldAbstractTpl or be "
        "polymorphic<ManifoldAbstractTpl>.");
    this->nx_ += _get_nx(c);
    this->ndx_ += _get_ndx(c);
    m_components.emplace_back(std::move(c));
  }

  inline void addComponent(const CartesianProductTpl &other) {
    for (const auto &c : other.m_components) {
      this->addComponent(c);
    }
  }

  explicit CartesianProductTpl()
      : Base(0, 0)
      , m_components() {}
  CartesianProductTpl(const CartesianProductTpl &) = default;
  CartesianProductTpl &operator=(const CartesianProductTpl &) = default;
  CartesianProductTpl(CartesianProductTpl &&) = default;
  CartesianProductTpl &operator=(CartesianProductTpl &&) = default;

  CartesianProductTpl(const std::vector<xyz::polymorphic<Base>> &components)
      : Base(0, 0)
      , m_components(components) {
    this->_calc_dims();
  }

  CartesianProductTpl(std::vector<xyz::polymorphic<Base>> &&components)
      : Base(0, 0)
      , m_components(std::move(components)) {
    this->_calc_dims();
  }

  CartesianProductTpl(std::initializer_list<xyz::polymorphic<Base>> components)
      : Base(0, 0)
      , m_components(components) {
    this->_calc_dims();
  }

  CartesianProductTpl(const xyz::polymorphic<Base> &left,
                      const xyz::polymorphic<Base> &right)
      : Base(left->nx() + right->nx(), left->ndx() + right->ndx())
      , m_components{left, right} {}

  bool isNormalized(const ConstVectorRef &x) const;

  template <class VectorType, class U = std::remove_const_t<VectorType>>
  std::vector<U> split_impl(VectorType &x) const;

  template <class VectorType, class U = std::remove_const_t<VectorType>>
  std::vector<U> split_vector_impl(VectorType &v) const;

  [[nodiscard]] std::vector<VectorRef> split(VectorRef x) const {
    return split_impl<VectorRef>(x);
  }

  [[nodiscard]] std::vector<ConstVectorRef>
  split(const ConstVectorRef &x) const {
    return split_impl<const ConstVectorRef>(x);
  }

  [[nodiscard]] std::vector<VectorRef> split_vector(VectorRef v) const {
    return split_vector_impl<VectorRef>(v);
  }

  [[nodiscard]] std::vector<ConstVectorRef>
  split_vector(const ConstVectorRef &v) const {
    return split_vector_impl<const ConstVectorRef>(v);
  }

  [[nodiscard]] VectorXs merge(const std::vector<VectorXs> &xs) const;

  [[nodiscard]] VectorXs merge_vector(const std::vector<VectorXs> &vs) const;

protected:
  std::vector<xyz::polymorphic<Base>> m_components;

  void _calc_dims() {
    this->nx_ = 0u;
    this->ndx_ = 0u;
    for (const auto &c : m_components) {
      this->nx_ += c->nx();
      this->ndx_ += c->ndx();
    }
  }

  template <class Concrete> static int _get_nx(const Concrete &c) {
    return c.nx();
  }
  template <class Concrete> static int _get_ndx(const Concrete &c) {
    return c.ndx();
  }
  static int _get_nx(const xyz::polymorphic<Base> &c) { return c->nx(); }
  static int _get_ndx(const xyz::polymorphic<Base> &c) { return c->ndx(); }

  void neutral_impl(VectorRef out) const;

  void rand_impl(VectorRef out) const;

  void integrate_impl(const ConstVectorRef &x, const ConstVectorRef &v,
                      VectorRef out) const;

  void difference_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                       VectorRef out) const;

  void Jintegrate_impl(const ConstVectorRef &x, const ConstVectorRef &v,
                       MatrixRef Jout, int arg) const;

  void JintegrateTransport_impl(const ConstVectorRef &x,
                                const ConstVectorRef &v, MatrixRef Jout,
                                int arg) const;

  void Jdifference_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                        MatrixRef Jout, int arg) const;
};

template <typename T>
auto operator*(const xyz::polymorphic<ManifoldAbstractTpl<T>> &left,
               const xyz::polymorphic<ManifoldAbstractTpl<T>> &right) {
  return CartesianProductTpl<T>(left, right);
}

template <typename T>
auto operator*(const CartesianProductTpl<T> &left,
               const xyz::polymorphic<ManifoldAbstractTpl<T>> &right) {
  CartesianProductTpl<T> out(left);
  out.addComponent(right);
  return out;
}

template <typename T>
auto operator*(const xyz::polymorphic<ManifoldAbstractTpl<T>> &left,
               const CartesianProductTpl<T> &right) {
  return right * left;
}

template <typename T>
auto operator*(const CartesianProductTpl<T> &left,
               const CartesianProductTpl<T> &right) {
  CartesianProductTpl<T> out{left};
  out.addComponent(right);
  return out;
}

} // namespace aligator

// implementation details
#include "aligator/modelling/spaces/cartesian-product.hxx"
