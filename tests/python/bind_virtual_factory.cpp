#include "proxddp/context.hpp"
#include "proxddp/python/visitors.hpp"
#include "proxddp/python/macros.hpp"

using std::shared_ptr;
using namespace proxddp::python;

struct MyVirtualData;

struct MyVirtualClass {

  MyVirtualClass() {}
  virtual ~MyVirtualClass() {}

  // take by shared_ptr
  virtual void doSomething(shared_ptr<MyVirtualData> const &data) const = 0;
  // take by reference
  virtual void doSomethingRef(MyVirtualData &data) const = 0;

  virtual shared_ptr<MyVirtualData> createData() const {
    return std::make_shared<MyVirtualData>();
  }
};

struct MyVirtualData {
  MyVirtualData() {}
  virtual ~MyVirtualData() {} // virtual dtor to mark class as polymorphic
};

auto callDoSomething(const MyVirtualClass &obj) {
  auto d = obj.createData();
  printf("Created MyVirtualData with address %p\n", (void *)d.get());
  obj.doSomething(d);
  return d;
}

auto callDoSomethingRef(const MyVirtualClass &obj) {
  auto d = obj.createData();
  printf("Created MyVirtualData with address %p\n", (void *)d.get());
  obj.doSomethingRef(*d);
  return d;
}

/// Wrapper classes
struct VirtualClassWrapper : MyVirtualClass, bp::wrapper<MyVirtualClass> {
  void doSomething(shared_ptr<MyVirtualData> const &data) const override {
    PROXDDP_PYTHON_OVERRIDE_PURE(void, "doSomething", data);
  }

  void doSomethingRef(MyVirtualData &data) const override {
    PROXDDP_PYTHON_OVERRIDE_PURE(void, "doSomethingRef", boost::ref(data));
  }

  shared_ptr<MyVirtualData> createData() const override {
    PROXDDP_PYTHON_OVERRIDE(shared_ptr<MyVirtualData>, MyVirtualClass,
                            createData, );
  }

  shared_ptr<MyVirtualData> default_createData() const {
    return MyVirtualClass::createData();
  }
};

BOOST_PYTHON_MODULE(bind_virtual_factory) {

  bp::register_ptr_to_python<shared_ptr<MyVirtualClass>>();
  bp::class_<VirtualClassWrapper, boost::noncopyable>(
      "MyVirtualClass", bp::init<>(bp::args("self")))
      .def("doSomething", bp::pure_virtual(&MyVirtualClass::doSomething),
           bp::args("self", "data"))
      .def("doSomethingRef", bp::pure_virtual(&MyVirtualClass::doSomethingRef),
           bp::args("self", "data"))
      .def(CreateDataPolymorphicPythonVisitor<MyVirtualClass,
                                              VirtualClassWrapper>());

  bp::class_<MyVirtualData, shared_ptr<MyVirtualData>, boost::noncopyable>(
      "MyVirtualData", bp::no_init)
      .def(bp::init<>(bp::args("self")))
      .def(PrintAddressVisitor<MyVirtualData>());

  bp::def("callDoSomething", callDoSomething, bp::args("obj"));
  bp::def("callDoSomethingRef", callDoSomethingRef, bp::args("obj"));
}
