#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 ARCHIVE.gpg [DESTINATION]" >&2
    exit 2
fi

archive=$1
destination=${2:-restored-session-markdown}
mkdir -p "$destination"
umask 077
gpg --decrypt "$archive" | zstd --decompress --stdout | tar -x -C "$destination"
echo "restored Markdown files to: $destination"
