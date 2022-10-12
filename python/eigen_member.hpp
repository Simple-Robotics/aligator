/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include <Eigen/Core>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/return_value_policy.hpp>

namespace proxddp {
namespace python {

namespace bp = boost::python;

namespace internal {

template <typename MatrixType, typename C> struct eigen_member {
public:
  eigen_member(MatrixType C::*which) : m_which(which) {}

  Eigen::Ref<MatrixType> operator()(C &c) const { return c.*m_which; }

  void operator()(C &c,
                  typename bp::detail::value_arg<MatrixType>::type d) const {
    c.*m_which = d;
  }

private:
  MatrixType C::*m_which;
};

} // namespace internal

template <class B, class MatrixType>
bp::object make_getter_eigen_ref(MatrixType B::*v) {
  typedef Eigen::Ref<MatrixType> RefType;
  return bp::make_function(internal::eigen_member<MatrixType, B>(v),
                           bp::return_value_policy<bp::return_by_value>(),
                           boost::mpl::vector2<RefType, B &>());
} // namespace internal

} // namespace python
} // namespace proxddp
