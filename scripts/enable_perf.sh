# Temporarily enable the kernel flags perf needs
set -x

sudo sysctl -w kernel/perf_event_paranoid=-1
sudo sysctl -w kernel/kptr_restrict=0
