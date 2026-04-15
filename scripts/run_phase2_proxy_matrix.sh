#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_phase2_proxy_matrix.sh \
    --sender-device-id cpu_fallback \
    --sender-backend cpu \
    --receiver-device-id server_remote \
    --upstream-host <receiver-ip>

This helper runs the phase-2 matrix in two stages:
  1) local actions once with network_profile=none
  2) remote actions once per proxy profile, automatically starting/stopping
     phase2_link_proxy.py for each profile

Common options:
  --sender-device-id <id>        Required. e.g. cpu_fallback, orin_nx_15w
  --sender-backend <backend>     Required. e.g. cpu or cuda:0
  --sender-device-profile <id>   Optional device profile name for preflight validation.
  --device-profiles-dir <dir>    Optional device profile directory.
  --allow-device-profile-mismatch Continue even if sender profile validation fails.
  --receiver-device-id <id>      Required for remote runs.
  --upstream-host <host>         Required for remote runs. Real receiver host.
  --upstream-port <port>         Default: 47001
  --profiles <csv>               Default: good,medium,poor
  --local-actions <csv>          Default: A0,A4
  --remote-actions <csv>         Default: A1,A2,A3
  --listen-host <host>           Default: 127.0.0.1
  --listen-port <port>           Default: 47002
  --proxy-seed <int>             Default: 123
  --proxy-enable-los-nlos        Enable LOS/NLOS stochastic channel model in proxy.
  --proxy-p-init-los <float>     Initial LOS probability. Default: 0.8
  --proxy-p-los-to-nlos <float>  LOS->NLOS transition prob. Default: 0.08
  --proxy-p-nlos-to-los <float>  NLOS->LOS transition prob. Default: 0.25
  --proxy-nlos-delay-scale <f>   NLOS delay multiplier. Default: 2.5
  --proxy-nlos-rate-scale <f>    NLOS rate multiplier. Default: 0.4
  --proxy-nlos-extra-jitter-ms <f> Extra jitter in NLOS. Default: 5.0
  --image-dir <dir>              Default: <repo>/data
  --output-root <dir>            Default: <repo>/outputs/phase2_execution/<sender>
  --proxy-log-dir <dir>          Default: <repo>/outputs/link_proxy/<sender>
  --max-images <n>               Default: 21
  --warmup <n>                   Default: 1
  --runs <n>                     Default: 3
  --skip-local                   Do not run the local-only stage.
  --skip-remote                  Do not run the remote/profile stage.
  --reference-detail-csv <path>  Optional override for remote reference CSV.

Examples:
  scripts/run_phase2_proxy_matrix.sh \
    --sender-device-id cpu_fallback \
    --sender-backend cpu \
    --receiver-device-id server_remote \
    --upstream-host <receiver-ip> \
    --remote-actions A3

  scripts/run_phase2_proxy_matrix.sh \
    --sender-device-id orin_nx_15w \
    --sender-backend cuda:0 \
    --receiver-device-id server_remote \
    --upstream-host <receiver-ip> \
    --profiles good,medium \
    --remote-actions A1,A2,A3
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SENDER_DEVICE_ID=""
SENDER_BACKEND=""
SENDER_DEVICE_PROFILE=""
DEVICE_PROFILES_DIR=""
ALLOW_DEVICE_PROFILE_MISMATCH="0"
RECEIVER_DEVICE_ID=""
UPSTREAM_HOST=""
UPSTREAM_PORT="47001"
PROFILES_CSV="good,medium,poor"
LOCAL_ACTIONS_CSV="A0,A4"
REMOTE_ACTIONS_CSV="A1,A2,A3"
LISTEN_HOST="127.0.0.1"
LISTEN_PORT="47002"
PROXY_SEED="123"
PROXY_ENABLE_LOS_NLOS="0"
PROXY_P_INIT_LOS="0.8"
PROXY_P_LOS_TO_NLOS="0.08"
PROXY_P_NLOS_TO_LOS="0.25"
PROXY_NLOS_DELAY_SCALE="2.5"
PROXY_NLOS_RATE_SCALE="0.4"
PROXY_NLOS_EXTRA_JITTER_MS="5.0"
IMAGE_DIR="$REPO_ROOT/data"
OUTPUT_ROOT=""
PROXY_LOG_DIR=""
REFERENCE_DETAIL_CSV=""
MAX_IMAGES="21"
WARMUP="1"
RUNS="3"
RUN_LOCAL="1"
RUN_REMOTE="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sender-device-id)
      SENDER_DEVICE_ID="$2"
      shift 2
      ;;
    --sender-backend)
      SENDER_BACKEND="$2"
      shift 2
      ;;
    --sender-device-profile)
      SENDER_DEVICE_PROFILE="$2"
      shift 2
      ;;
    --device-profiles-dir)
      DEVICE_PROFILES_DIR="$2"
      shift 2
      ;;
    --allow-device-profile-mismatch)
      ALLOW_DEVICE_PROFILE_MISMATCH="1"
      shift
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
    --listen-host)
      LISTEN_HOST="$2"
      shift 2
      ;;
    --listen-port)
      LISTEN_PORT="$2"
      shift 2
      ;;
    --proxy-seed)
      PROXY_SEED="$2"
      shift 2
      ;;
    --proxy-enable-los-nlos)
      PROXY_ENABLE_LOS_NLOS="1"
      shift
      ;;
    --proxy-p-init-los)
      PROXY_P_INIT_LOS="$2"
      shift 2
      ;;
    --proxy-p-los-to-nlos)
      PROXY_P_LOS_TO_NLOS="$2"
      shift 2
      ;;
    --proxy-p-nlos-to-los)
      PROXY_P_NLOS_TO_LOS="$2"
      shift 2
      ;;
    --proxy-nlos-delay-scale)
      PROXY_NLOS_DELAY_SCALE="$2"
      shift 2
      ;;
    --proxy-nlos-rate-scale)
      PROXY_NLOS_RATE_SCALE="$2"
      shift 2
      ;;
    --proxy-nlos-extra-jitter-ms)
      PROXY_NLOS_EXTRA_JITTER_MS="$2"
      shift 2
      ;;
    --image-dir)
      IMAGE_DIR="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --proxy-log-dir)
      PROXY_LOG_DIR="$2"
      shift 2
      ;;
    --reference-detail-csv)
      REFERENCE_DETAIL_CSV="$2"
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
    --skip-local)
      RUN_LOCAL="0"
      shift
      ;;
    --skip-remote)
      RUN_REMOTE="0"
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

if [[ -z "$SENDER_DEVICE_ID" || -z "$SENDER_BACKEND" ]]; then
  echo "--sender-device-id and --sender-backend are required." >&2
  exit 1
fi

if [[ "$RUN_REMOTE" == "1" && ( -z "$RECEIVER_DEVICE_ID" || -z "$UPSTREAM_HOST" ) ]]; then
  echo "--receiver-device-id and --upstream-host are required for remote runs." >&2
  exit 1
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="$REPO_ROOT/outputs/phase2_execution/$SENDER_DEVICE_ID"
fi
if [[ -z "$PROXY_LOG_DIR" ]]; then
  PROXY_LOG_DIR="$REPO_ROOT/outputs/link_proxy/$SENDER_DEVICE_ID"
fi
if [[ -z "$REFERENCE_DETAIL_CSV" ]]; then
  REFERENCE_DETAIL_CSV="$OUTPUT_ROOT/none/phase2_detail.csv"
fi

mkdir -p "$OUTPUT_ROOT" "$PROXY_LOG_DIR"

split_csv() {
  local csv="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<< "$csv"
}

run_python_suite() {
  local network_profile="$1"
  local output_dir="$2"
  shift 2
  python scripts/run_phase2_execution_suite.py \
    --sender-device-id "$SENDER_DEVICE_ID" \
    --sender-backend "$SENDER_BACKEND" \
    "${PROFILE_ARGS[@]}" \
    --network-profile "$network_profile" \
    --image-dir "$IMAGE_DIR" \
    --output-dir "$output_dir" \
    --max-images "$MAX_IMAGES" \
    --warmup "$WARMUP" \
    --runs "$RUNS" \
    "$@"
}

PROFILE_ARGS=()
if [[ -n "$SENDER_DEVICE_PROFILE" ]]; then
  PROFILE_ARGS+=(--sender-device-profile "$SENDER_DEVICE_PROFILE")
fi
if [[ -n "$DEVICE_PROFILES_DIR" ]]; then
  PROFILE_ARGS+=(--device-profiles-dir "$DEVICE_PROFILES_DIR")
fi
if [[ "$ALLOW_DEVICE_PROFILE_MISMATCH" == "1" ]]; then
  PROFILE_ARGS+=(--allow-device-profile-mismatch)
fi

PROXY_PID=""

cleanup_proxy() {
  if [[ -n "${PROXY_PID:-}" ]]; then
    kill "$PROXY_PID" >/dev/null 2>&1 || true
    wait "$PROXY_PID" >/dev/null 2>&1 || true
    PROXY_PID=""
  fi
}

trap cleanup_proxy EXIT

wait_for_proxy() {
  local attempts=50
  local idx
  for ((idx=0; idx<attempts; idx+=1)); do
    if [[ -n "${PROXY_PID:-}" ]] && kill -0 "$PROXY_PID" >/dev/null 2>&1; then
      sleep 0.2
      return 0
    fi
    sleep 0.2
  done
  return 1
}

LOCAL_ACTIONS=()
REMOTE_ACTIONS=()
PROFILES=()
split_csv "$LOCAL_ACTIONS_CSV" LOCAL_ACTIONS
split_csv "$REMOTE_ACTIONS_CSV" REMOTE_ACTIONS
split_csv "$PROFILES_CSV" PROFILES

echo "[run_phase2_proxy_matrix] sender=$SENDER_DEVICE_ID backend=$SENDER_BACKEND"
if [[ -n "$SENDER_DEVICE_PROFILE" ]]; then
  echo "[run_phase2_proxy_matrix] sender_profile=$SENDER_DEVICE_PROFILE"
fi
echo "[run_phase2_proxy_matrix] local_actions=${LOCAL_ACTIONS[*]}"
echo "[run_phase2_proxy_matrix] remote_actions=${REMOTE_ACTIONS[*]}"
echo "[run_phase2_proxy_matrix] profiles=${PROFILES[*]}"
echo "[run_phase2_proxy_matrix] output_root=$OUTPUT_ROOT"
if [[ "$PROXY_ENABLE_LOS_NLOS" == "1" ]]; then
  echo "[run_phase2_proxy_matrix] proxy_los_nlos=enabled p_init_los=$PROXY_P_INIT_LOS p_los_to_nlos=$PROXY_P_LOS_TO_NLOS p_nlos_to_los=$PROXY_P_NLOS_TO_LOS nlos_delay_scale=$PROXY_NLOS_DELAY_SCALE nlos_rate_scale=$PROXY_NLOS_RATE_SCALE nlos_extra_jitter_ms=$PROXY_NLOS_EXTRA_JITTER_MS"
fi

if [[ "$RUN_LOCAL" == "1" ]]; then
  echo "[run_phase2_proxy_matrix] running local stage"
  run_python_suite \
    "none" \
    "$OUTPUT_ROOT/none" \
    --actions "${LOCAL_ACTIONS[@]}"
fi

if [[ "$RUN_REMOTE" == "1" ]]; then
  if [[ ! -f "$REFERENCE_DETAIL_CSV" ]]; then
    echo "Reference detail CSV not found: $REFERENCE_DETAIL_CSV" >&2
    exit 1
  fi

  for profile in "${PROFILES[@]}"; do
    cleanup_proxy
    log_path="$PROXY_LOG_DIR/${profile}.jsonl"
    output_dir="$OUTPUT_ROOT/${profile}_proxy"
    echo "[run_phase2_proxy_matrix] starting proxy profile=$profile log=$log_path"
    proxy_cmd=(
      python scripts/phase2_link_proxy.py
      --listen-host "$LISTEN_HOST"
      --listen-port "$LISTEN_PORT"
      --upstream-host "$UPSTREAM_HOST"
      --upstream-port "$UPSTREAM_PORT"
      --profile "$profile"
      --seed "$PROXY_SEED"
      --log-jsonl "$log_path"
      --verbose
    )
    if [[ "$PROXY_ENABLE_LOS_NLOS" == "1" ]]; then
      proxy_cmd+=(
        --enable-los-nlos
        --p-init-los "$PROXY_P_INIT_LOS"
        --p-los-to-nlos "$PROXY_P_LOS_TO_NLOS"
        --p-nlos-to-los "$PROXY_P_NLOS_TO_LOS"
        --nlos-delay-scale "$PROXY_NLOS_DELAY_SCALE"
        --nlos-rate-scale "$PROXY_NLOS_RATE_SCALE"
        --nlos-extra-jitter-ms "$PROXY_NLOS_EXTRA_JITTER_MS"
      )
    fi
    "${proxy_cmd[@]}" &
    PROXY_PID="$!"

    if ! wait_for_proxy; then
      echo "Proxy did not become ready for profile=$profile" >&2
      exit 1
    fi

    echo "[run_phase2_proxy_matrix] running remote stage profile=$profile"
    run_python_suite \
      "$profile" \
      "$output_dir" \
      --receiver-device-id "$RECEIVER_DEVICE_ID" \
      --remote-host "$LISTEN_HOST" \
      --remote-port "$LISTEN_PORT" \
      --reference-detail-csv "$REFERENCE_DETAIL_CSV" \
      --actions "${REMOTE_ACTIONS[@]}"
  done
fi

cleanup_proxy
echo "[run_phase2_proxy_matrix] done"
