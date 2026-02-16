#!/bin/bash
# scripts/run.sh — Start the server
cd "$(dirname "$0")/.."
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
