#!/usr/bin/env bash
# Phase 1 of the 260503_RPi_full_backup import.
# - Skips overlapping exp dirs (uses cp -n / rsync --ignore-existing)
# - Logs every src->dst pair to a log file
# - Does NOT touch the 4 mismatched-overlap exp dirs (Phase 2 decision)
# - Does NOT touch zip archives, master.zip, pigpio-master, __pycache__,
#   or the explicitly-legacy "Author2 failed" / "z_old_outdated" subdirs.

set -euo pipefail

BACKUP=<REPO_ROOT>/260503_RPi_full_backup/latent_maze-RPI
WORKTREE=<REPO_ROOT>
RPI_CODE_DST=$WORKTREE/code/RPi_latMaz_exp_code
MAZE_DST=$WORKTREE/data_in/mazes
EXP_DST=$WORKTREE/data_in/0_raw_exp_dirs_from_RPi

LOG=$WORKTREE/data_transfer_cld/c$(date +%y%m%d-%H%M%S)_RPi_backup_import.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "[phase1 RPi-backup import] started $(date)"
echo "  backup root:   $BACKUP"
echo "  worktree:      $WORKTREE"
echo

# --- 1.1 Exp scripts + utils + config + display.sh + checklist ---
echo "[1.1] Copying RPi exp scripts and root files to $RPI_CODE_DST"
cp -n "$BACKUP"/*.py            "$RPI_CODE_DST"/ 2>/dev/null || true
cp -n "$BACKUP"/*.txt           "$RPI_CODE_DST"/ 2>/dev/null || true
cp -n "$BACKUP"/*.sh            "$RPI_CODE_DST"/ 2>/dev/null || true
cp -n "$BACKUP"/checklist       "$RPI_CODE_DST"/ 2>/dev/null || true
echo "  scripts now in dst: $(ls "$RPI_CODE_DST"/*.py 2>/dev/null | wc -l) .py files"
echo

# --- 1.2 test_exp_dir fixture ---
echo "[1.2] Copying test_exp_dir/"
cp -Rn "$BACKUP/test_exp_dir" "$RPI_CODE_DST"/ 2>/dev/null || true
echo

# --- 1.3 Maze files (copy only ones not already present) ---
echo "[1.3] Copying NEW maze files to $MAZE_DST"
n_before=$(ls "$MAZE_DST" 2>/dev/null | wc -l)
cp -n "$BACKUP/data_in"/*  "$MAZE_DST"/ 2>/dev/null || true
n_after=$(ls "$MAZE_DST" 2>/dev/null | wc -l)
echo "  maze files: $n_before -> $n_after (added $((n_after - n_before)))"
echo

# --- 1.4 Exp data_out dirs — only the 140 backup-only dirs ---
echo "[1.4] Copying NEW exp dirs to $EXP_DST (backup-only; existing untouched)"
added=0
skipped=0
for src in "$BACKUP/data_out"/*; do
  [ -d "$src" ] || continue
  name=$(basename "$src")
  dst=$EXP_DST/$name
  if [ -e "$dst" ]; then
    skipped=$((skipped + 1))
    continue
  fi
  cp -R "$src" "$dst"
  added=$((added + 1))
done
echo "  exp dirs added: $added; skipped (already exists): $skipped"
echo
echo "[phase1 RPi-backup import] finished $(date)"
echo "  log: $LOG"
