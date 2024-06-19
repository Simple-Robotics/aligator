/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include <eigenpy/fwd.hpp>
#include <fmt/format.h>

#include "aligator/python/fwd.hpp"

namespace aligator {
namespace python {
namespace bp = boost::python;

// fwd-declaration
bp::arg operator""_a(const char *argname, std::size_t);

template <typename T>
struct ClonePythonVisitor : bp::def_visitor<ClonePythonVisitor<T>> {
  template <typename PyT> void visit(PyT &obj) const {
    obj.def("clone", &T::clone, bp::args("self"), "Clone the object.");
  }
};

template <typename T>
struct CreateDataPythonVisitor : bp::def_visitor<CreateDataPythonVisitor<T>> {
  template <typename Pyclass> void visit(Pyclass &obj) const {
    using ReturnPtrType = decltype(std::declval<T>().createData());
    ReturnPtrType (T::*createData)(const context::CommonModelDataContainer &)
        const = &T::createData;
    obj.def("createData", createData, bp::args("self", "container"),
            "Create a data object.");
  }
};

template <typename T, typename TWrapper>
struct ConfigurePythonVisitor
    : bp::def_visitor<ConfigurePythonVisitor<T, TWrapper>> {
  template <typename Pyclass> void visit(Pyclass &obj) const {
    obj.def("configure", &T::configure, &TWrapper::default_configure,
            bp::args("self", "container"), "Create and configure CommonModel.");
  }
};

/// Visitor for exposing a polymorphic factory for data classes that can be
/// overridden from Python.
/// @sa CreateDataPythonVisitor
/// @sa StageFunctionTpl
/// @tparam T The wrapped class
/// @tparam TWrapper The wrapper class
template <typename T, typename TWrapper>
struct CreateDataPolymorphicPythonVisitor
    : bp::def_visitor<CreateDataPolymorphicPythonVisitor<T, TWrapper>> {
  template <typename PyClass> void visit(PyClass &obj) const {
    using ReturnPtrType = decltype(std::declval<T>().createData());
    ReturnPtrType (T::*createData)(const context::CommonModelDataContainer &)
        const = &T::createData;
    obj.def("createData", createData, &TWrapper::default_createData,
            bp::args("self", "container"), "Create a data object.");
  }
};

template <typename T>
struct CopyableVisitor : bp::def_visitor<CopyableVisitor<T>> {
  template <typename PyClass> void visit(PyClass &obj) const {
    obj.def("copy", &copy, bp::arg("self"), "Returns a copy of this.");
  }

private:
  static T copy(const T &self) { return T(self); }
};

template <typename T>
struct PrintableVisitor : bp::def_visitor<PrintableVisitor<T>> {
  template <typename PyClass> void visit(PyClass &obj) const {
    obj.def(bp::self_ns::str(bp::self)).def(bp::self_ns::repr(bp::self));
  }
};

template <typename T>
struct PrintAddressVisitor : bp::def_visitor<PrintAddressVisitor<T>> {
  template <typename PyClass> void visit(PyClass &obj) const {
    obj.def("printAddress", printAddress, bp::args("self"));
  }
  static void *getAddress(const T &a) { return (void *)&a; }
  static void printAddress(const T &a) {
    fmt::print("Address: {:p}\n", getAddress(a));
  }
};

// fwd declaration
template <typename SolverType> struct SolverVisitor;

} // namespace python
} // namespace aligator
