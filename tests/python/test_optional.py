import bind_optional


def test_none_if_zero():
    x = bind_optional.none_if_zero(0)
    y = bind_optional.none_if_zero(-1)
    assert x is None
    assert y == -1


def test_struct_ctors():
    # test struct ctors

    struct = bind_optional.mystruct()
    assert struct.a is None
    assert struct.b is None
    assert struct.msg == "i am struct"

    ## no 2nd arg automatic overload using bp::optional
    struct = bind_optional.mystruct(2)
    assert struct.a == 2
    assert struct.b is None

    struct = bind_optional.mystruct(13, -1.0)
    assert struct.a == 13
    assert struct.b == -1.0


def test_struct_setters():
    struct = bind_optional.mystruct()
    struct.a = 1
    assert struct.a == 1

    struct.b = -3.14
    assert struct.b == -3.14

    # set to None
    struct.a = None
    struct.b = None
    struct.msg = None
    assert struct.a is None
    assert struct.b is None
    assert struct.msg is None


def test_factory():
    struct = bind_optional.create_if_true(False)
    assert struct is None
    struct = bind_optional.create_if_true(True)
    assert struct.a == 0
    assert struct.b is None


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(sys.argv))
