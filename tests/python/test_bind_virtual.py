import bind_virtual_factory as bvf
import pytest


class ImplClass(bvf.MyVirtualClass):
    def __init__(self):
        self.val = 42
        super().__init__()

    def createData(self):
        return ImplData(self)

    # override MyVirtualClass::doSomething(data)
    def doSomething(self, data: bvf.MyVirtualData):
        print("Hello from doSomething!")
        data.printAddress()
        assert isinstance(data, ImplData)
        print("Data value:", data.value)
        data.value += 1

    def doSomethingRef(self, data: bvf.MyVirtualData):
        print("Hello from doSomethingRef!")
        data.printAddress()
        assert isinstance(data, ImplData)
        print("Data value:", data.value)


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
    d1 = bvf.callDoSomething(obj)
    print("Output data.value:", d1.value)


def test_call_do_something_ref():
    obj = ImplClass()
    print("Ref variant:")
    with pytest.raises(AssertionError) as e_info:  # error *must* be raised
        bvf.callDoSomethingRef(obj)
        print(e_info)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
