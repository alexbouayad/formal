#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/src/formal/grammar/tree_sitter"

uv pip list | grep '^tree-sitter-' | while read -r REPOSITORY VERSION; do
	rm -rf "$REPOSITORY"

	git clone --no-checkout --filter=blob:none --depth 1 "https://github.com/tree-sitter/$REPOSITORY.git"
	git -C "$REPOSITORY" fetch --depth 1 origin tag "v$VERSION"
	git -C "$REPOSITORY" sparse-checkout set --no-cone tree-sitter.json grammar.js scanner.h '**/src/grammar.json' '**/src/scanner.c' '**/src/scanner.cc' '**/src/scanner.cxx' '**/src/*.h' '**/src/*.hpp' '**/src/tree_sitter'
	git -C "$REPOSITORY" switch --detach "tags/v$VERSION"
	rm -rf "$REPOSITORY/.git"

	find "$REPOSITORY" -type f \( -name "scanner.c" -o -name "scanner.cc" -o -name "scanner.cxx" \) | while read -r SCANNER_FILE; do
		SRC_DIR="$(dirname "$SCANNER_FILE")"

		if [[ "$SCANNER_FILE" == *.c ]]; then
			gcc -shared -I"$SRC_DIR" -o "$SRC_DIR/scanner.so" -fPIC "$SCANNER_FILE"
		else
			g++ -shared -I"$SRC_DIR" -o "$SRC_DIR/scanner.so" -fPIC "$SCANNER_FILE"
		fi
	done
done
