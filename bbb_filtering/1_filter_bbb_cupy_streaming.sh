#!/bin/bash
set -euo pipefail

# Configuration
INPUT_DIR="$HOME/inputs_round2"            # Folder with compressed files (~30GB each)
OUTPUT_DIR="$HOME/outputs_round2"          # Final outputs folder (initially empty)
SCRIPT="$HOME/scripts/filter_bbb_cupy_streaming.py"  # Python processing script

# Unified tmpfs mount for both compartments
MOUNT_POINT="/tmp/pipeline_ram"
TMPFS_SIZE="70G" 

# SHM_ORIG is where chunks are written and SHM_DIR is where the script expects files
SHM_ORIG="$MOUNT_POINT/shm"
SHM_DIR="/dev/shm/pipeline_chunks"

LOG_DIR="$HOME/logs"
LOG_FILE="$LOG_DIR/pipeline_round2.log"

# Each chunk is 15GB (in bytes)
CHUNK_SIZE_BYTES=$((15 * 1024 * 1024 * 1024))
# Using dd with bs=1M, number of blocks is about 20480
CHUNK_COUNT=$((CHUNK_SIZE_BYTES / (1024 * 1024)))

# Limit: Total allowed chunks (across working space) is 4.
MAX_TOTAL_CHUNKS=4
PARALLEL_JOBS=3              # Always run 3 Python processes concurrently

# Define glob patterns:
# CHUNK_PATTERN1: unprocessed chunk file (e.g. filename_00)
# CHUNK_PATTERN2: file marked as processing (e.g. filename_00.processing)
CHUNK_PATTERN1="*_[0-9][0-9]"
CHUNK_PATTERN2="*_[0-9][0-9].processing"

# Logging & Error Handling
log() {
  echo "[$(date +"%F %T")] $*" | tee -a "$LOG_FILE"
}

error_exit() {
  log "ERROR: $*"
  exit 1
}

trap 'error_exit "Error at line $LINENO: $BASH_COMMAND"' ERR

# Setup Directories & Bind Mounts
log "SETUP: Starting directory and mount setup..."
if mountpoint -q "$MOUNT_POINT"; then
  log "SETUP: Unmounting existing mount at $MOUNT_POINT"
  sudo umount -R "$MOUNT_POINT" || error_exit "SETUP: Failed to unmount $MOUNT_POINT"
fi

rm -rf "$MOUNT_POINT" "$OUTPUT_DIR"
mkdir -p "$MOUNT_POINT" "$OUTPUT_DIR" "$LOG_DIR"

log "SETUP: Mounting $MOUNT_POINT as tmpfs with a size limit of $TMPFS_SIZE."
sudo mount -t tmpfs -o size=$TMPFS_SIZE tmpfs "$MOUNT_POINT"

mkdir -p "$SHM_ORIG"

if mountpoint -q "$SHM_DIR"; then
  log "SETUP: Unmounting existing SHM_DIR at $SHM_DIR"
  sudo umount "$SHM_DIR" || error_exit "SETUP: Failed to unmount $SHM_DIR"
fi
sudo mkdir -p "$SHM_DIR"
sudo mount --bind "$SHM_ORIG" "$SHM_DIR"

# Check that working space is on the same device
shm_dev=$(stat -c '%d' "$SHM_DIR")
log "SETUP: Device number for SHM_DIR ($SHM_DIR): $shm_dev"
log "SETUP: SHM_DIR set to $SHM_DIR (bind mount from $SHM_ORIG)."

# Utility Functions
get_total_chunks() {
  # Count only non-empty files
  find "$SHM_DIR" -maxdepth 1 -type f \( -name "$CHUNK_PATTERN1" -o -name "$CHUNK_PATTERN2" \) 2>/dev/null | wc -l
}

log_chunk_counts() {
  total=$(get_total_chunks)
  log "STATUS: Chunk counts in SHM_DIR: ${total}"
}

cleanup_empty_chunks() {
  # Delete any file matching chunk patterns that is 0 bytes in size
  find "$SHM_DIR" -maxdepth 1 -type f \( -name "$CHUNK_PATTERN1" -o -name "$CHUNK_PATTERN2" \) -size 0 -delete
}

wait_for_space() {
  local total
  # First, clean up any empty files that might be stale
  cleanup_empty_chunks
  total=$(get_total_chunks)
  while [ "$total" -ge "$MAX_TOTAL_CHUNKS" ]; do
    log "WAIT: Total chunks in memory ($total) reached limit ($MAX_TOTAL_CHUNKS). Waiting..."
    sleep 1
    cleanup_empty_chunks
    total=$(get_total_chunks)
  done
}

# Helper Function: Launch Python Jobs
launch_jobs() {
  while [ "$(jobs -pr | wc -l)" -lt "$PARALLEL_JOBS" ]; do
    next_chunk=$(find "$SHM_DIR" -maxdepth 1 -type f -name "$CHUNK_PATTERN1" | sort | head -n 1 || true)
    if [ -z "$next_chunk" ]; then
      log "LAUNCH: No unprocessed chunk in SHM_DIR."
      break
    fi
    # Check if the chunk is non-empty before processing
    if [ ! -s "$next_chunk" ]; then
      log "LAUNCH: Found empty chunk $(basename "$next_chunk"), removing it."
      rm -f "$next_chunk"
      continue
    fi

    processing_chunk="${next_chunk}.processing"
    log "LAUNCH: Renaming $(basename "$next_chunk") to $(basename "$processing_chunk")"
    mv "$next_chunk" "$processing_chunk"
    base=$(basename "$processing_chunk" .processing)
    out_file="$OUTPUT_DIR/${base}.cxsmiles.lz4"
    log "LAUNCH: Launching Python process for $base"
    python3 -u "$SCRIPT" "$processing_chunk" "$out_file" >> "$LOG_FILE" 2>&1 &
    pid=$!
    log "LAUNCH: Python process launched with PID $pid for $base"
    sleep 0.5
    log_chunk_counts
  done
}

# Main Processing Loop
for input_file in "$INPUT_DIR"/*.cxsmiles.bz2; do
  log "MAIN: Starting processing of input file: $input_file"
  chunk_index=0

  # Decompress the input file using lbzip2 and split into 15GB chunks
  (
    log "DECOMP: Starting decompression of $input_file" >&2
    lbzip2 -dc "$input_file" || true
    log "DECOMP: Finished decompression stream for $input_file" >&2
  ) | while true; do
    wait_for_space

    base_name=$(basename "$input_file" .cxsmiles.bz2)
    out_chunk="$SHM_DIR/${base_name}_$(printf "%02d" "$chunk_index")"
    log "MAIN: Writing chunk $(printf "%02d" "$chunk_index") to $out_chunk"
    
    if ! dd bs=1M count="$CHUNK_COUNT" iflag=fullblock of="$out_chunk" status=none; then
      if [ -s "$out_chunk" ]; then
        log "MAIN: Accepted final partial chunk: $(basename "$out_chunk") ($(stat -c%s "$out_chunk") bytes)"
        chunk_index=$((chunk_index+1))
      else
        log "MAIN: Removing empty chunk $out_chunk"
        rm -f "$out_chunk"
      fi
      break
    fi

    if [ ! -s "$out_chunk" ]; then
      log "MAIN: Created chunk $(printf "%02d" "$chunk_index") is empty. Removing..."
      rm -f "$out_chunk"
      break
    fi

    log "MAIN: Created chunk $(printf "%02d" "$chunk_index"): $(stat -c%s "$out_chunk") bytes"
    chunk_index=$((chunk_index+1))

    launch_jobs
    log_chunk_counts
  done

  log "MAIN: Decompression of $input_file complete. Waiting for remaining chunks..."
  while [ -n "$(find "$SHM_DIR" -maxdepth 1 -type f \( -name "$CHUNK_PATTERN1" -o -name "$CHUNK_PATTERN2" \) 2>/dev/null)" ]; do
    launch_jobs
    log_chunk_counts
    sleep 2
  done
  log "MAIN: Finished processing $input_file"
done

# Final Merging Step
final_output="$OUTPUT_DIR/final_filtered.cxsmiles.lz4"
log "FINAL: Merging processed outputs into $final_output"
if compgen -G "$OUTPUT_DIR"/*".cxsmiles.lz4" > /dev/null; then
  cat "$OUTPUT_DIR"/*".cxsmiles.lz4" > "$final_output"
  log "FINAL: Final output merged successfully."
else
  error_exit "FINAL: No processed chunks found. Check processing steps."
fi

log "FINAL: Processing completed successfully."
