#!/usr/bin/env bash
#
# Run the test suite on a remote host. The working tree is streamed over SSH
# (no rsync dependency) into ~/<REMOTE_DIR>, then the requested make target runs
# there.
#
# Usage:
#   bash run-tests.sh <host> [make-target] [KEY=VALUE ...]
#
# Examples:
#   bash run-tests.sh debby
#   bash run-tests.sh debby check
#   bash run-tests.sh debby test-real-parity
#   bash run-tests.sh debby test-real-bench QWEN38_BENCH_CONTEXT=128k
#   REMOTE_DIR=qwen-reasoning-trainer-test bash run-tests.sh debby test
#
# The optional KEY=VALUE arguments are exported for the remote make invocation.
# The small WikiText-2 evaluation corpus is synced when present; models and
# other datasets stay on the remote host.
set -euo pipefail

HOST="${1:?usage: bash run-tests.sh <host> [make-target] [KEY=VALUE ...]}"
TARGET="${2:-check}"
shift 2 2>/dev/null || shift $#
REMOTE_DIR="${REMOTE_DIR:-qwen-reasoning-trainer}"

ENV_ARGS=()
for arg in "$@"; do
	case "$arg" in
	*=*) ENV_ARGS+=("$arg") ;;
	*) echo "run-tests.sh: expected KEY=VALUE, got $arg" >&2; exit 2 ;;
	esac
done

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "syncing $ROOT -> $HOST:~/$REMOTE_DIR"
ssh "$HOST" "mkdir -p ~/$REMOTE_DIR"
tar \
	--exclude=.git \
	--exclude=build \
	--exclude=bin \
	--exclude=models \
	--exclude=datasets \
	--exclude=source/e2e-tests/.venv \
	--exclude='*.gguf' \
	--exclude='*.part' \
	--exclude='*.tmp' \
	-czf - . | ssh "$HOST" "tar -xzf - -C ~/$REMOTE_DIR"

# datasets/ is excluded above, but the evaluation corpus is small and required
# for the real-model quality tests. Sync just that file.
if [ -f datasets/wikitext2-test.txt ]; then
	echo "syncing datasets/wikitext2-test.txt"
	ssh "$HOST" "mkdir -p ~/$REMOTE_DIR/datasets && cat > ~/$REMOTE_DIR/datasets/wikitext2-test.txt" \
		< datasets/wikitext2-test.txt
fi

echo "running make $TARGET on $HOST"
ENV_CMD=""
if [ "${#ENV_ARGS[@]}" -gt 0 ]; then
	printf -v ENV_CMD '%q ' "${ENV_ARGS[@]}"
fi
ssh "$HOST" "cd ~/$REMOTE_DIR && env $ENV_CMD make $TARGET"
