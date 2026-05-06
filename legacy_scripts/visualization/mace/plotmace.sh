#!/bin/bash

GP="$HOME/scripts/plot_mace_live.gp"

if [ $# -lt 1 ] || [ $# -gt 4 ]; then
  echo "Usage: plotmace <mace_train.log> [window_epochs] [refresh_seconds] [mode]"
  echo "  mode: always | onchange   (default: always)"
  exit 1
fi

FILE="$1"
WINDOW="${2:-100}"
REFRESH="${3:-5}"
MODE="${4:-always}"

if [ ! -f "$FILE" ]; then
  echo "Error: file not found: $FILE"
  exit 1
fi

if ! [[ "$WINDOW" =~ ^[0-9]+$ ]] || [ "$WINDOW" -lt 1 ]; then
  echo "Error: window_epochs must be a positive integer"
  exit 1
fi

if ! [[ "$REFRESH" =~ ^[0-9]+$ ]] || [ "$REFRESH" -lt 1 ]; then
  echo "Error: refresh_seconds must be a positive integer"
  exit 1
fi

if [[ "$MODE" != "always" && "$MODE" != "onchange" ]]; then
  echo "Error: mode must be 'always' or 'onchange'"
  exit 1
fi

if [ ! -f "$GP" ]; then
  echo "Error: gnuplot script not found: $GP"
  exit 1
fi

get_mtime() {
  if stat -f %m "$1" >/dev/null 2>&1; then
    stat -f %m "$1"
  else
    stat -c %Y "$1"
  fi
}

prev_mtime=0

while true; do
  if [ "$MODE" = "onchange" ]; then
    curr_mtime=$(get_mtime "$FILE" 2>/dev/null)
    if [ "$curr_mtime" != "$prev_mtime" ]; then
      clear
      gnuplot -e "file='$FILE'; win=$WINDOW" "$GP"
      prev_mtime="$curr_mtime"
    fi
  else
    clear
    gnuplot -e "file='$FILE'; win=$WINDOW" "$GP"
  fi

  sleep "$REFRESH"
done