#!/usr/bin/env bash
#
# Run the test suite on a remote host. The working tree is streamed over SSH
# (no rsync dependency) into ~/<REMOTE_DIR>, then the requested make target runs
# there.
#
# Usage:
#   bash run-tests.sh <host> [make-target]
#
# Examples:
#   bash run-tests.sh debby
#   bash run-tests.sh debby check
#   REMOTE_DIR=qwen-reasoning-trainer-test bash run-tests.sh debby test
#
set -euo pipefail

HOST="${1:?usage: bash run-tests.sh <host> [make-target]}"
TARGET="${2:-check}"
REMOTE_DIR="${REMOTE_DIR:-qwen-reasoning-trainer}"

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

echo "running make $TARGET on $HOST"
ssh "$HOST" "cd ~/$REMOTE_DIR && make $TARGET"
