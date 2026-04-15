#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_phase2_nvpmodel_study.sh \
    --upstream-host <receiver-ip> \
    --receiver-device-id server_remote

This helper iterates over multiple sender nvpmodel/device profiles and runs
the existing phase-2 proxy matrix for each mode into a dedicated output folder.

Common options:
  --modes <csv>                  Default: orin_nx_maxn,orin_nx_40w,orin_nx_20w,orin_nx_15w,orin_nx_10w
  --sender-backend <backend>     Default: cuda:0
  --receiver-device-id <id>      Required for remote runs.
  --upstream-host <host>         Required for remote runs.
  --upstream-port <port>         Default: 47001
  --suite-root <dir>             Default: <repo>/outputs/phase2_execution_nvpmodel_real
  --proxy-log-root <dir>         Default: <repo>/outputs/link_proxy_nvpmodel_real
  --profiles <csv>               Default: good,medium,poor
  --local-actions <csv>          Default: A0,A4
  --remote-actions <csv>         Default: A1,A2,A3
  --max-images <n>               Default: 21
  --warmup <n>                   Default: 1
  --runs <n>                     Default: 3
  --use-jetson-clocks            Also apply jetson_clocks after switching nvpmodel.
  --skip-local                   Do not run local-only stage.
  --skip-remote                  Do not run remote/profile stage.
  --allow-device-profile-mismatch Continue if sender profile validation fails.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODES_CSV="orin_nx_maxn,orin_nx_40w,orin_nx_20w,orin_nx_15w,orin_nx_10w"
SENDER_BACKEND="cuda:0"
RECEIVER_DEVICE_ID=""
UPSTREAM_HOST=""
UPSTREAM_PORT="47001"
SUITE_ROOT="$REPO_ROOT/outputs/phase2_execution_nvpmodel_real"
PROXY_LOG_ROOT="$REPO_ROOT/outputs/link_proxy_nvpmodel_real"
PROFILES_CSV="good,medium,poor"
LOCAL_ACTIONS_CSV="A0,A4"
REMOTE_ACTIONS_CSV="A1,A2,A3"
MAX_IMAGES="21"
WARMUP="1"
RUNS="3"
RUN_LOCAL="1"
RUN_REMOTE="1"
USE_JETSON_CLOCKS="0"
ALLOW_DEVICE_PROFILE_MISMATCH="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --modes)
      MODES_CSV="$2"
      shift 2
      ;;
    --sender-backend)
      SENDER_BACKEND="$2"
      shift 2
      ;;
    --receiver-device-id)
      RECEIVER_DEVICE_ID="$2"
      shift 2
      ;;
    --upstream-host)
      UPSTREAM_HOST="$2"
      shift 2
      ;;
    --upstream-port)
      UPSTREAM_PORT="$2"
      shift 2
      ;;
    --suite-root)
      SUITE_ROOT="$2"
      shift 2
      ;;
    --proxy-log-root)
      PROXY_LOG_ROOT="$2"
      shift 2
      ;;
    --profiles)
      PROFILES_CSV="$2"
      shift 2
      ;;
    --local-actions)
      LOCAL_ACTIONS_CSV="$2"
      shift 2
      ;;
    --remote-actions)
      REMOTE_ACTIONS_CSV="$2"
      shift 2
      ;;
    --max-images)
      MAX_IMAGES="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --runs)
      RUNS="$2"
      shift 2
      ;;
    --use-jetson-clocks)
      USE_JETSON_CLOCKS="1"
      shift
      ;;
    --skip-local)
      RUN_LOCAL="0"
      shift
      ;;
    --skip-remote)
      RUN_REMOTE="0"
      shift
      ;;
    --allow-device-profile-mismatch)
      ALLOW_DEVICE_PROFILE_MISMATCH="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$RUN_REMOTE" == "1" && ( -z "$RECEIVER_DEVICE_ID" || -z "$UPSTREAM_HOST" ) ]]; then
  echo "--receiver-device-id and --upstream-host are required for remote runs." >&2
  exit 1
fi

mkdir -p "$SUITE_ROOT" "$PROXY_LOG_ROOT"

IFS=',' read -r -a MODES <<< "$MODES_CSV"

for mode in "${MODES[@]}"; do
  mode="$(echo "$mode" | xargs)"
  if [[ -z "$mode" ]]; then
    continue
  fi

  echo "================================================================================"
  echo "[run_phase2_nvpmodel_study] mode=$mode sender_backend=$SENDER_BACKEND"
  echo "[run_phase2_nvpmodel_study] suite_root=$SUITE_ROOT"
  echo "================================================================================"

  ctl_cmd=(
    python scripts/device_profile_ctl.py
    --profile "$mode"
    --sender-backend "$SENDER_BACKEND"
    --apply
  )
  if [[ "$USE_JETSON_CLOCKS" == "1" ]]; then
    ctl_cmd+=(--use-jetson-clocks)
  fi
  "${ctl_cmd[@]}"

  run_cmd=(
    scripts/run_phase2_proxy_matrix.sh
    --sender-device-id "$mode"
    --sender-device-profile "$mode"
    --sender-backend "$SENDER_BACKEND"
    --profiles "$PROFILES_CSV"
    --local-actions "$LOCAL_ACTIONS_CSV"
    --remote-actions "$REMOTE_ACTIONS_CSV"
    --output-root "$SUITE_ROOT/$mode"
    --proxy-log-dir "$PROXY_LOG_ROOT/$mode"
    --max-images "$MAX_IMAGES"
    --warmup "$WARMUP"
    --runs "$RUNS"
  )
  if [[ "$RUN_REMOTE" == "1" ]]; then
    run_cmd+=(
      --receiver-device-id "$RECEIVER_DEVICE_ID"
      --upstream-host "$UPSTREAM_HOST"
      --upstream-port "$UPSTREAM_PORT"
    )
  fi
  if [[ "$RUN_LOCAL" == "0" ]]; then
    run_cmd+=(--skip-local)
  fi
  if [[ "$RUN_REMOTE" == "0" ]]; then
    run_cmd+=(--skip-remote)
  fi
  if [[ "$ALLOW_DEVICE_PROFILE_MISMATCH" == "1" ]]; then
    run_cmd+=(--allow-device-profile-mismatch)
  fi

  "${run_cmd[@]}"
done
