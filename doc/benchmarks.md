# Benchmarks

We use [google benchmark](https://github.com/google/benchmark/tree/v1.5.0) to define C++ benchmarks
which are able to aggregate data from runs, and Flame Graphs to produce a breakdown of the various function calls
and their importance as a proportion of the call stack.

Here's Crocoddyl's flame graph:
![croc-talos-arm](images/flamegraph-croc.svg)
Here's for `aligator::SolverFDDP`:
![prox-talos-arm](images/flamegraph-prox.svg)
