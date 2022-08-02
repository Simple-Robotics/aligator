# Developer's guide

When creating the CMake build, make sure to add the `-DCMAKE_EXPORT_COMPILE_COMMANDS=1` flag. See its documentation [here](https://cmake.org/cmake/help/latest/variable/CMAKE_EXPORT_COMPILE_COMMANDS.html).

## Debugging a C++ executable

This project builds some C++ examples and tests. Debugging them is fairly straightforward using GDB:

```bash
gdb path/to/executable
```

with the appropriate command line arguments. Examples will appear in the binaries of `build/examples`. Make sure to look at GDB's documentation.

If you want to catch `std::exception` instances thrown, enter the following command once in GDB:

```gdb
(gdb) catch throw std::exception
```

## Debugging a Python example or test

In order for debug symbols to be loaded and important variables not being optimized out, you will want to compile in `DEBUG` mode.

Then, you can run the module under `gdb` using

```bash
gdb --args python example/file.py
```

If you want to look at Eigen types such as vectors and matrices, you should look into the [`eigengdb`](https://github.com/dmillard/eigengdb) plugin for GDB.

## Hybrid debugging with Visual Studio Code

**TODO** Finish documenting this
