# Use the brendangregg/FlameGraph repo with perl scripts
set -x

cd build

CMD=./examples/example-croc-talos-arm
CMD_OPTS=""

FLAMEGRAPH_DIR=${FLAMEGRAPH_DIR:-${HOME}/git-repos/FlameGraph}
if [ ! -d $FLAMEGRAPH_DIR ]
then
    echo "Supplied FLAMEGRAPH_DIR=${FLAMEGRAPH_DIR} is not a directory"
    return 1
fi

ST_COLL="$FLAMEGRAPH_DIR/stackcollapse-perf.pl"
FLAMEGRAPH_CMD="$FLAMEGRAPH_DIR/flamegraph.pl"

# using --call-graph dwarf means you can record perf w/o using
# -fno-omit-frame-pointer compiler option
perf record --call-graph dwarf $CMD $CMD_OPTS
perf script | $ST_COLL > out.folded
$FLAMEGRAPH_CMD out.folded > out.svg
rm out.folded
