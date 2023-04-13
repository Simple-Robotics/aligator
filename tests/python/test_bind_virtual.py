import bind_virtual_factory as bvf
import pytest


class ImplClass(bvf.MyVirtualClass):
    def __init__(self):
        self.val = 42
        super().__init__()

    def createData(self):
        return ImplData(self)

    # override MyVirtualClass::doSomethingPtr(shared_ptr data)
    def doSomethingPtr(self, data: bvf.MyVirtualData):
        print("Hello from doSomething!")
        data.printAddress()
        assert isinstance(data, ImplData)
        print("Data value:", data.value)
        data.value += 1

    # override MyVirtualClass::doSomethingPtr(data&)
    def doSomethingRef(self, data: bvf.MyVirtualData):
        print("Hello from doSomethingRef!")
        data.printAddress()
        print(type(data))
        assert isinstance(data, ImplData)
        print("Data value:", data.value)
        data.value += 1


class ImplData(bvf.MyVirtualData):
    def __init__(self, c: ImplClass):
        super().__init__()
        self.value = c.val


def test_instantiate_child():
    obj = ImplClass()
    data = obj.createData()
    data.printAddress()


def test_call_do_something_ptr():
    obj = ImplClass()
    print("Calling doSomething (by ptr)")
    d1 = bvf.callDoSomethingPtr(obj)
    print("Output data.value:", d1.value)


def test_call_do_something_ref():
    obj = ImplClass()
    print("Ref variant:")
    d2 = bvf.callDoSomethingRef(obj)
    print(d2.value)
    print("-----")


def test_iden_fns():

    obj = ImplClass()
    d = obj.createData()
    print(d, type(d))

    # take and return const T&
    d1 = bvf.iden_ref(d)
    print(d1, type(d1))
    assert isinstance(d1, ImplData)

    # take a shared_ptr, return const T&
    d2 = bvf.iden_shared(d)
    assert isinstance(d2, ImplData)
    print(d2, type(d2))

    print("copy shared ptr -> py -> cpp")
    d3 = bvf.copy_shared(d)
    assert isinstance(d3, ImplData)
    print(d3, type(d3))


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
