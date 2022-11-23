cd build
BENCHMARK_REPS=40
BENCHMARK_FLAGS="--benchmark_repetitions=$BENCHMARK_REPS --benchmark_display_aggregates_only=true"
./examples/example-croc-talos-arm $BENCHMARK_FLAGS
