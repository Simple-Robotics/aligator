#include "proxddp/context.hpp"
#include "proxddp/python/visitors.hpp"
#include "proxddp/python/macros.hpp"

using std::shared_ptr;
using namespace proxddp::python;

// fwd declaration
struct MyVirtualData;

struct MyVirtualClass {

  MyVirtualClass() {}
  virtual ~MyVirtualClass() {}

  // polymorphic fn taking arg by shared_ptr
  virtual void doSomethingPtr(shared_ptr<MyVirtualData> const &data) const = 0;
  // polymorphic fn taking arg by reference
  virtual void doSomethingRef(MyVirtualData &data) const = 0;

  virtual shared_ptr<MyVirtualData> createData() const {
    return std::make_shared<MyVirtualData>();
  }
};

struct MyVirtualData {
  MyVirtualData() {}
  virtual ~MyVirtualData() {} // virtual dtor to mark class as polymorphic
};

auto callDoSomethingPtr(const MyVirtualClass &obj) {
  auto d = obj.createData();
  printf("Created MyVirtualData with address %p\n", (void *)d.get());
  obj.doSomethingPtr(d);
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

  void doSomethingPtr(shared_ptr<MyVirtualData> const &data) const override {
    PROXDDP_PYTHON_OVERRIDE_PURE(void, "doSomethingPtr", data);
  }

  void doSomethingRef(MyVirtualData &data) const override {
    PROXDDP_PYTHON_OVERRIDE_PURE(
        void, "doSomethingRef",
        boost::ref(data)); // ref() required otherwise Boost.python assumes
                           // to-value conversion
  }

  shared_ptr<MyVirtualData> createData() const override {
    PROXDDP_PYTHON_OVERRIDE(shared_ptr<MyVirtualData>, MyVirtualClass,
                            createData, );
  }

  shared_ptr<MyVirtualData> default_createData() const {
    return MyVirtualClass::createData();
  }
};

/// this trampoline class does nothing but is *ABSOLUTELY* required to ensure
/// downcasting works properly with non-smart ptr signatures because otherwise,
/// there is no handle to the original PyObject* every single polymorphic type
/// exposed to Python should be exposed thru a trampoline
struct DataWrapper : MyVirtualData, bp::wrapper<MyVirtualData> {};

/// Take and return a const ref
const MyVirtualData &iden_ref(const MyVirtualData &d) {
  // try cast to holder
  return d;
}

/// Take a shared_ptr, return by const ref
const MyVirtualData &iden_shared(const shared_ptr<MyVirtualData> &d) {
  // get boost.python's custom deleter
  // boost.python hides the handle to the original object in there
  // dter being nonzero indicates shared_ptr was wrapped by Boost.Python
  auto *dter = std::get_deleter<bp::converter::shared_ptr_deleter>(d);
  if (dter != 0)
    printf("> input shared_ptr has a deleter\n");
  return *d;
}

/// Take and return a shared_ptr
shared_ptr<MyVirtualData> copy_shared(const shared_ptr<MyVirtualData> &d) {
  auto *dter = std::get_deleter<bp::converter::shared_ptr_deleter>(d);
  if (dter != 0)
    printf("> input shared_ptr has a deleter\n");
  return d;
}

BOOST_PYTHON_MODULE(bind_virtual_factory) {

  assert(std::is_polymorphic<MyVirtualClass>::value &&
         "MyVirtualClass should be polymorphic!");
  assert(std::is_polymorphic<MyVirtualData>::value &&
         "MyVirtualData should be polymorphic!");

  bp::class_<VirtualClassWrapper, boost::noncopyable>(
      "MyVirtualClass", bp::init<>(bp::args("self")))
      .def("doSomething", bp::pure_virtual(&MyVirtualClass::doSomethingPtr),
           bp::args("self", "data"))
      .def("doSomethingRef", bp::pure_virtual(&MyVirtualClass::doSomethingRef),
           bp::args("self", "data"))
      .def(CreateDataPolymorphicPythonVisitor<MyVirtualClass,
                                              VirtualClassWrapper>());

  bp::register_ptr_to_python<shared_ptr<MyVirtualData>>();
  /// trampoline used as 1st argument
  /// otherwise if passed as "HeldType", we need to define
  /// the constructor and call initializer manually
  bp::class_<DataWrapper, boost::noncopyable>("MyVirtualData", bp::no_init)
      .def(bp::init<>(bp::args("self")))
      .def(PrintAddressVisitor<MyVirtualData>());

  bp::def("callDoSomethingPtr", callDoSomethingPtr, bp::args("obj"));
  bp::def("callDoSomethingRef", callDoSomethingRef, bp::args("obj"));

  bp::def("iden_ref", iden_ref, bp::return_internal_reference<>());
  bp::def("iden_shared", iden_shared, bp::return_internal_reference<>());
  bp::def("copy_shared", copy_shared);
}
