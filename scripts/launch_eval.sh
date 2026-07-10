#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-main}"
LIMIT="${2:-256}"

fm-mapb evaluate --experiment "${EXPERIMENT}" --limit "${LIMIT}"
