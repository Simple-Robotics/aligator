set -x


cd build
CMD=./examples/example-croc-talos-arm
FILTER=prox
REPS=10
# CMD_OPTS="--benchmark_filter=$FILTER"
CMD_OPTS="--benchmark_repetitions=$REPS"

BUILD_DIR=$PWD
FLAMEGRAPH_DIR=$HOME/git-repos/FlameGraph

ST_COLL="$FLAMEGRAPH_DIR/stackcollapse-perf.pl"
FLAMEGRAPH_CMD="$FLAMEGRAPH_DIR/flamegraph.pl"

# using --call-graph dwarf means you can record perf w/o using
# -fno-omit-frame-pointer compiler option
perf record --call-graph dwarf $CMD $CMD_OPTS
perf script | $ST_COLL > out.folded
$FLAMEGRAPH_CMD out.folded > out.svg

