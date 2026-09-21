#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
ROOT="${BIRDPILOT_ROOT:-/mnt/sdc/Stasis/birdpilot}"
PHYSICAL_GPU="${BIRDPILOT_PHYSICAL_GPU:-3}"
CLEAN_CONFIG="configs/exp014_clean_seed42.json"
AUG_CONFIG="configs/exp014_augmented_seed42.json"

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
mkdir -p outputs/robustness_runs outputs/l20_evidence outputs/l20_logs

assert_cuda() {
  python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; assert torch.cuda.device_count() == 1, torch.cuda.device_count(); print("visible GPU:", torch.cuda.get_device_name(0), "torch:", torch.__version__, "CUDA:", torch.version.cuda)'
}

run_to_epoch() {
  local config="$1"
  local run_name="$2"
  local target="$3"
  local run_dir="outputs/robustness_runs/${run_name}"
  local log="outputs/l20_logs/${run_name}.log"
  local -a resume_args=()
  if [[ -f "${run_dir}/metrics.jsonl" ]]; then
    local last_epoch
    last_epoch="$(python -c 'import json,sys; rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]; print(rows[-1]["epoch"])' "${run_dir}/metrics.jsonl")"
    if (( last_epoch >= target )); then
      echo "${run_name} already reached epoch ${last_epoch}; target ${target}, skipping"
      return 0
    fi
    [[ -f "${run_dir}/last.pt" ]] || { echo "missing resume checkpoint: ${run_dir}/last.pt" >&2; return 1; }
    resume_args=(--resume "${run_dir}/last.pt")
  elif [[ -e "${run_dir}" ]]; then
    echo "refusing ambiguous existing run directory without metrics: ${run_dir}" >&2
    return 1
  fi
  python src/train_robustness.py --config "${config}" --stop-after-epoch "${target}" "${resume_args[@]}" 2>&1 | tee -a "${log}"
}

show_status() {
  python - <<'PY'
import json
from pathlib import Path
for seed in (42, 43, 44):
  for arm in ("clean", "augmented"):
    name = f"exp014_{arm}_seed{seed}"
    path = Path("outputs/robustness_runs") / name / "metrics.jsonl"
    if not path.is_file():
        print(name, "NOT_STARTED")
        continue
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    row = rows[-1]
    print(name, "epoch", row["epoch"], "clean", f'{row["clean_validation_accuracy"]:.6f}', "robust", f'{row["robust_condition_mean_accuracy"]:.6f}', "eligible", row["eligible"], "seconds", f'{row["elapsed_seconds"]:.1f}')
PY
}

select_seed() {
  local seed="$1"
  local final="outputs/robustness_runs/exp014_seed${seed}_selection.json"
  local temporary="${final}.tmp.$$"
  python src/select_robustness_candidate.py \
    --baseline-summary outputs/exp013_m0_robustness_full_v2/summary.json \
    --clean-metrics "outputs/robustness_runs/exp014_clean_seed${seed}/metrics.jsonl" \
    --augmented-metrics "outputs/robustness_runs/exp014_augmented_seed${seed}/metrics.jsonl" \
    --output "${temporary}"
  mv -f "${temporary}" "${final}"
}

select_all() {
  for seed in 42 43 44; do
    select_seed "${seed}"
  done
  local final="outputs/robustness_runs/exp014_repeated_selection.json"
  local temporary="${final}.tmp.$$"
  python src/summarize_repeated_robustness.py \
    --selection 42=outputs/robustness_runs/exp014_seed42_selection.json \
    --selection 43=outputs/robustness_runs/exp014_seed43_selection.json \
    --selection 44=outputs/robustness_runs/exp014_seed44_selection.json \
    --output "${temporary}"
  mv -f "${temporary}" "${final}"
}

case "${MODE}" in
  probe)
    assert_cuda
    stamp="$(date +%Y%m%dT%H%M%S)"
    python src/probe_training_device.py --config "${AUG_CONFIG}" --output "outputs/l20_evidence/device_probe_${stamp}.json"
    ;;
  gate)
    assert_cuda
    run_to_epoch "${CLEAN_CONFIG}" exp014_clean_seed42 3
    run_to_epoch "${AUG_CONFIG}" exp014_augmented_seed42 3
    show_status
    ;;
  resume)
    assert_cuda
    run_to_epoch "${CLEAN_CONFIG}" exp014_clean_seed42 10
    run_to_epoch "${AUG_CONFIG}" exp014_augmented_seed42 10
    show_status
    ;;
  complete)
    assert_cuda
    for seed in 42 43 44; do
      run_to_epoch "configs/exp014_clean_seed${seed}.json" "exp014_clean_seed${seed}" 10
      run_to_epoch "configs/exp014_augmented_seed${seed}.json" "exp014_augmented_seed${seed}" 10
    done
    show_status
    select_all
    ;;
  status)
    show_status
    ;;
  select)
    select_all
    ;;
  *)
    echo "usage: $0 {probe|gate|resume|complete|status|select}" >&2
    exit 2
    ;;
esac
