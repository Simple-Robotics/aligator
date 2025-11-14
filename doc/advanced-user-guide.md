# Developer and advanced user guide

When creating the CMake build, make sure to add the `-DCMAKE_EXPORT_COMPILE_COMMANDS=1` flag. See its documentation [here](https://cmake.org/cmake/help/latest/variable/CMAKE_EXPORT_COMPILE_COMMANDS.html).

A template project for using **aligator** with CMake and C++ can be found in the [aligator-cmake-example-project](https://github.com/Simple-Robotics/aligator-cmake-example-project) repository.

## Creating a Python extension module

When **aligator** is installed, the CMake configuration file (`aligatorConfig.cmake`) provides a CMake function to help users easily create a [Python extension module](https://docs.python.org/3/extending/extending.html).
Users can write an extension module in C++ for performance reasons when providing e.g. custom constraints, cost functions, dynamics, and so on.

The CMake function is called as follows:
```{.cmake}
aligator_create_python_extension(<name> [WITH_SOABI] <sources...>)
```

This will create a CMake `MODULE` target named `<name>` on which the user can set properties and add an `install` directive.

An usage example can be found in [this repo](https://github.com/Simple-Robotics/aligator-cmake-example-project).

## Debugging

### Debugging a C++ executable

This project builds some C++ examples and tests. Debugging them is fairly straightforward using GDB:

```bash
gdb path/to/executable
```

with the appropriate command line arguments. Examples will appear in the binaries of `build/examples`. Make sure to look at GDB's documentation.

If you want to catch `std::exception` instances thrown, enter the following command once in GDB:

```gdb
(gdb) catch throw std::exception
```

### Debugging a Python example or test

In order for debug symbols to be loaded and important variables not being optimized out, you will want to compile in `DEBUG` mode.

Then, you can run the module under `gdb` using

```bash
gdb --args python example/file.py
```

If you want to look at Eigen types such as vectors and matrices, you should look into the [`eigengdb`](https://github.com/dmillard/eigengdb) plugin for GDB.

### Hybrid debugging with Visual Studio Code

[TODO]

## Using **aligator**'s parallelization features

The `SolverProxDDP` solver is able to leverage multicore CPU architectures.

### Inside your code

Before calling the solver make sure to enable parallelization as follows:

In Python:

```python
solver.rollout_type = aligator.ROLLOUT_LINEAR
solver.linear_solver_choice = aligator.LQ_SOLVER_PARALLEL
solver.setNumThreads(num_threads)
```

And in C++:
```cpp
std::size_t num_threads = 4ul;  // for example
solver.rollout_type = aligator::RolloutType::LINEAR;
solver.linear_solver_choice = aligator::LQSolverChoice::PARALLEL;
solver.setNumThreads(num_threads);
```

### Shell setup for CPU core optimization
**Aligator** uses OpenMP for parallelization which is setup using environment variables in your shell. The settings are local to your shell.

#### Visualization
Printing OpenMP parameters at launch:
```bash
export OMP_DISPLAY_ENV=VERBOSE
```
Print when a thread is launched and with which affinity (CPU thread(s) on where it will try to run):
```bash
export OMP_DISPLAY_AFFINITY=TRUE
```

#### Core and thread assignment
OpenMP operates with **places** which define a CPU thread or core reserved for a thread. **Places** can be a CPU thread or an entire CPU core (which can have one thread, or multiple with hyperthreading).

##### Assigning places with CPU threads:
```bash
export OMP_PLACES ="threads(n)" # Threads will run on the first nth CPU threads, with one thread per CPU thread.
```
or
```bash
export OMP_PLACES="{0},{1},{2}" # Threads will run on CPU threads 0, 1 ,2
```
##### Assigning places with CPU cores:

Threads will run on the first nth CPU cores, with one thread per core, even if the core has multiple threads
```bash
export OMP_PLACES="cores(n)"
```

For more info on places see [here](https://www.ibm.com/docs/en/xl-fortran-linux/16.1.0?topic=openmp-omp-places).

##### Using only performance cores (Intel performance hybrid architectures)

Some modern CPUs have a mix of performance (P) and efficiency (E) cores. The E-cores are often slower, hence we should
have OpenMP schedule threads on P-cores only.

Get your CPU model with
```bash
lscpu | grep -i "Model Name"
```
Get CPU core info with:
```bash
lscpu -e

# with an i7-13800H
CPU NODE SOCKET CORE L1d:L1i:L2:L3 ONLINE    MAXMHZ   MINMHZ      MHZ
  0    0      0    0 0:0:0:0          yes 5000.0000 400.0000  400.000
  1    0      0    0 0:0:0:0          yes 5000.0000 400.0000  400.000
  2    0      0    1 4:4:1:0          yes 5000.0000 400.0000  400.000
  3    0      0    1 4:4:1:0          yes 5000.0000 400.0000  400.000
  4    0      0    2 8:8:2:0          yes 5200.0000 400.0000  400.000
  5    0      0    2 8:8:2:0          yes 5200.0000 400.0000 5176.303
  6    0      0    3 12:12:3:0        yes 5200.0000 400.0000 1482.743
  7    0      0    3 12:12:3:0        yes 5200.0000 400.0000  400.000
  8    0      0    4 16:16:4:0        yes 5000.0000 400.0000 3485.561
  9    0      0    4 16:16:4:0        yes 5000.0000 400.0000  721.684
 10    0      0    5 20:20:5:0        yes 5000.0000 400.0000 1641.311
 11    0      0    5 20:20:5:0        yes 5000.0000 400.0000  400.000
 12    0      0    6 24:24:6:0        yes 4000.0000 400.0000  400.000
 13    0      0    7 25:25:6:0        yes 4000.0000 400.0000 2949.734
 14    0      0    8 26:26:6:0        yes 4000.0000 400.0000 2554.695
 15    0      0    9 27:27:6:0        yes 4000.0000 400.0000 3588.623
 16    0      0   10 28:28:7:0        yes 4000.0000 400.0000  400.000
 17    0      0   11 29:29:7:0        yes 4000.0000 400.0000  400.000
 18    0      0   12 30:30:7:0        yes 4000.0000 400.0000  400.000
 19    0      0   13 31:31:7:0        yes 4000.0000 400.0000 3610.068
```
A little digging on the internet tells us that this CPU has 6 performance cores and 8 efficiency cores for a total of 20 threads. We see higher frequencies in core 0 to 5: these are the performance cores. To use only performance cores on this CPU you would set:
```bash
export OMP_PLACES="cores(6)"
# or
export OMP_PLACES="threads(12)"
```
> [!IMPORTANT]
> Put your PC in performance mode (usually found in the power settings).

## Profiling

We use [google benchmark](https://github.com/google/benchmark/tree/v1.5.0) to define C++ benchmarks
which are able to aggregate data from runs, and [Flame Graphs](https://github.com/brendangregg/FlameGraph) to produce a breakdown of the various function calls and their importance as a proportion of the call stack.

If you have the Rust toolchain and `cargo` installed, we suggest you install [cargo-flamegraph](https://github.com/flamegraph-rs/flamegraph). Then, you can create a flame graph with the following command:

```bash
flamegraph -o my_flamegraph.svg -- ./build/examples/example-croc-talos-arm
```


Here's Crocoddyl's flame graph:
![croc-talos-arm](images/flamegraph-croc.svg)
Here's for `aligator::SolverFDDP`:
![prox-talos-arm](images/flamegraph-prox.svg)
