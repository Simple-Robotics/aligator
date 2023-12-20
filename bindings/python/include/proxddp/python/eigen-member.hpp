/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include <Eigen/Core>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/return_value_policy.hpp>

#include <fmt/ostream.h>

namespace proxddp {
namespace python {

namespace bp = boost::python;

namespace internal {

/// @brief Functor which wraps a pointer-to-data-member and will return an \ref
/// Eigen::Ref when called.
/// @see   \ref boost/python/data_members.hpp (struct member)
template <typename MatrixType, typename Class> struct eigen_member {
public:
  typedef Eigen::Ref<const MatrixType> ConstMatRef;

  eigen_member(MatrixType Class::*which) : m_which(which) {}

  Eigen::Ref<MatrixType> operator()(Class &c) const { return c.*m_which; }

  void operator()(Class &c,
                  typename bp::detail::value_arg<ConstMatRef>::type d) const {
    c.*m_which = d;
  }

private:
  MatrixType Class::*m_which;
};

} // namespace internal

/// @brief    Create a getter for Eigen::Matrix type objects which returns an
/// Eigen::Ref.
/// @see      \ref boost/python/data_member.hpp (\ref make_getter(D C::*pm,
/// ...))
template <class C, class MatrixType>
bp::object make_getter_eigen_matrix(MatrixType C::*v) {
  typedef Eigen::Ref<MatrixType> RefType;
  return bp::make_function(
      internal::eigen_member<MatrixType, C>(v),
      bp::return_value_policy<bp::return_by_value,
                              bp::with_custodian_and_ward_postcall<0UL, 1UL>>(),
      boost::mpl::vector2<RefType, C &>());
}

template <class C, class MatrixType, class Policies>
bp::object make_setter_eigen_matrix(MatrixType C::*v,
                                    Policies const &policies) {
  typedef Eigen::Ref<const MatrixType> ConstRefType;
  return bp::make_function(
      internal::eigen_member<MatrixType, C>(v), policies,
      boost::mpl::vector3<void, C &, const ConstRefType>());
}

template <class C, class MatrixType>
bp::object make_setter_eigen_matrix(MatrixType C::*v) {
  return make_setter_eigen_matrix(v, bp::default_call_policies());
}

} // namespace python
} // namespace proxddp
