#!/bin/bash
# arc_organize.sh — Organize logs and checkpoints on ARC login node.
#
# Scans $HOME recursively for:
#   Logs        : *.out, *.err, gpu_usage_*.log
#   Checkpoints : *.pth, *.pt, *.ckpt
#
# Moves them into:
#   ~/src/logs/
#   ~/src/checkpoints/
#
# Naming conflicts resolved by appending _1, _2, ... before the extension.
# Uploaded and run by total_setup.py.
#
# Usage:
#   bash arc_organize.sh
#   bash arc_organize.sh --dry-run

set -euo pipefail

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

LOG_DIR="$HOME/src/logs"
CKPT_DIR="$HOME/src/checkpoints"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

# ── Conflict-safe move ──────────────────────────────────────────────────────
move_with_conflict() {
    local src="$1"
    local dest_dir="$2"
    local base
    base=$(basename "$src")
    local ext="${base##*.}"
    local stem="${base%.*}"
    local dest="$dest_dir/$base"
    local n=1

    while [ -f "$dest" ]; do
        dest="$dest_dir/${stem}_${n}.${ext}"
        n=$(( n + 1 ))
    done

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [dry-run] $src  ->  $dest"
    else
        mv "$src" "$dest"
        echo "  MOVE: $base  ->  $(basename "$dest")"
    fi
}

# ── Find helper — excludes dest dirs and hidden/env dirs ───────────────────
arc_find() {
    # $@ = -name patterns joined with -o
    find "$HOME" \
        -not -path "$LOG_DIR/*" \
        -not -path "$CKPT_DIR/*" \
        -not -path "$HOME/.conda/*" \
        -not -path "$HOME/.cache/*" \
        -not -path "$HOME/.local/*" \
        -not -path "$HOME/miniforge3/*" \
        -not -path "$HOME/scratch/*" \
        -type f \( "$@" \) \
        -print0
}

# ── Logs ───────────────────────────────────────────────────────────────────
echo "=============================="
echo "LOGS  ->  $LOG_DIR"
echo "=============================="

log_count=0
while IFS= read -r -d '' f; do
    move_with_conflict "$f" "$LOG_DIR"
    log_count=$(( log_count + 1 ))
done < <(arc_find -name "*.out" -o -name "*.err" -o -name "gpu_usage_*.log")

[ "$log_count" -eq 0 ] && echo "  Nothing to move."

# ── Checkpoints ────────────────────────────────────────────────────────────
echo ""
echo "=============================="
echo "CHECKPOINTS  ->  $CKPT_DIR"
echo "=============================="

ckpt_count=0
while IFS= read -r -d '' f; do
    move_with_conflict "$f" "$CKPT_DIR"
    ckpt_count=$(( ckpt_count + 1 ))
done < <(arc_find -name "*.pth" -o -name "*.pt" -o -name "*.ckpt")

[ "$ckpt_count" -eq 0 ] && echo "  Nothing to move."

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "=============================="
if [ "$DRY_RUN" = "1" ]; then
    echo "  [dry-run] Would move: $log_count log(s), $ckpt_count checkpoint(s)."
else
    echo "  Moved: $log_count log(s), $ckpt_count checkpoint(s)."
fi
echo "=============================="
