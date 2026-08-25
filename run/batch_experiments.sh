#!/bin/bash
set -e

PROJECT_ROOT="/home/oliver/NEW_GPS/GPS-SLAM"
CONFIGS_DIR="$PROJECT_ROOT/configs/experiments"
OUTPUT_BASE="$PROJECT_ROOT/output/experiments"
SLAM_TRAINER="$PROJECT_ROOT/build/slam_trainer"
ATE_SCRIPT="$PROJECT_ROOT/scripts/ate_general.py"
GT_POSE_DIR="$PROJECT_ROOT/data/replica/office0/camera"
BASE_CONFIG="$PROJECT_ROOT/configs/release/replica/office0_tracking_halfres_fixed.yaml"

mkdir -p "$CONFIGS_DIR" "$OUTPUT_BASE"

SUMMARY="$OUTPUT_BASE/summary.csv"
echo "name,status,fps,fusion_ms,local_opt_ms,gs_num,valid_frames,total_frames,ate_cm" > "$SUMMARY"

run_exp() {
    local name="$1"
    shift
    local overrides="$@"
    local cfg="$CONFIGS_DIR/${name}.yaml"
    local out_dir="$OUTPUT_BASE/$name"

    echo ""
    echo "========================================"
    echo "Running: $name"
    echo "Overrides: $overrides"
    echo "========================================"

    # Generate config with Python
    python3 - "$BASE_CONFIG" "$cfg" "$out_dir" $overrides <<'PYEOF'
import sys, yaml
base_path, out_path, ws_dir = sys.argv[1], sys.argv[2], sys.argv[3]
overrides = sys.argv[4:]
with open(base_path) as f:
    cfg = yaml.safe_load(f)
cfg['workspace_dir'] = ws_dir
for o in overrides:
    path, val = o.split('=', 1)
    keys = path.split('/')
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    # Try to preserve type
    try:
        val = int(val)
    except ValueError:
        try:
            val = float(val)
        except ValueError:
            pass
    node[keys[-1]] = val
with open(out_path, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
PYEOF

    rm -rf "$out_dir"

    # Run experiment
    if ! "$SLAM_TRAINER" "$cfg" > "$out_dir.log" 2>&1; then
        echo "$name crashed"
        echo "$name,CRASH,0,0,0,0,0,0,N/A" >> "$SUMMARY"
        return
    fi

    # Parse timing
    time_log="$out_dir/time_log.txt"
    fps=0; fusion_ms=0; local_opt_ms=0; gs_num=0
    if [ -f "$time_log" ]; then
        line=$(grep "\[PIPELINE AVG TIME\]" "$time_log" | tail -1)
        fps=$(echo "$line" | grep -oP 'FPS: \K[0-9.]+' || echo 0)
        fusion_ms=$(echo "$line" | grep -oP 'per frame fusion time: \K[0-9.]+' || echo 0)
        local_opt_ms=$(echo "$line" | grep -oP 'localOptimize time: \K[0-9.]+' || echo 0)
        gs_num=$(echo "$line" | grep -oP 'GS num: \K[0-9]+' || echo 0)
    fi

    # Check poses and compute ATE
    pose_dir="$out_dir/val/pose"
    valid_frames=0; total_frames=0; ate_cm="N/A"
    if [ -d "$pose_dir" ]; then
        total_frames=$(ls "$pose_dir"/frame*.txt 2>/dev/null | wc -l)
        valid_frames=$(python3 -c "
import numpy as np, os
n=0
for f in sorted(os.listdir('$pose_dir')):
    if f.endswith('.txt'):
        p=np.loadtxt(os.path.join('$pose_dir',f))
        if not (np.isnan(p).any() or np.isinf(p).any()): n+=1
print(n)
" 2>/dev/null || echo 0)

        if [ "$valid_frames" -eq "$total_frames" ] && [ "$total_frames" -gt 0 ]; then
            ate_out=$(python3 "$ATE_SCRIPT" --gt_path "$GT_POSE_DIR" --est_path "$pose_dir" 2>&1)
            ate_cm=$(echo "$ate_out" | grep -oP 'ATE actual RMSE: \K[0-9.]+' || echo "N/A")
        fi
    fi

    status="OK"
    if [ "$valid_frames" -ne "$total_frames" ] || [ "$total_frames" -eq 0 ]; then
        status="TRACK_LOST"
    fi

    echo "$name,$status,$fps,$fusion_ms,$local_opt_ms,$gs_num,$valid_frames,$total_frames,$ate_cm" >> "$SUMMARY"
    echo "Result: $name | Status=$status | FPS=$fps | ATE=$ate_cm | GS=$gs_num"
}

# Run all experiments
run_exp exp_01_baseline
run_exp exp_02_iters_10         PIPE/local_opt_iters=10
run_exp exp_03_iters_5          PIPE/local_opt_iters=5
run_exp exp_04_interval_10      PIPE/local_opt_interval=10
run_exp exp_05_sh_1             MODEL/sh_degree=1
run_exp exp_06_sh_0             MODEL/sh_degree=0
run_exp exp_07_voxel_016        PIPE/TSDF/voxel_size=0.016 PIPE/TSDF/trunc_dist=0.064
run_exp exp_08_kf_max_4         PIPE/keyframe_select_max=4
run_exp exp_09_local_win_1      PIPE/localframe_cam_window_length=1
run_exp exp_10_combo_fast       PIPE/local_opt_iters=10 PIPE/local_opt_interval=10 PIPE/keyframe_select_max=4 MODEL/sh_degree=1
run_exp exp_11_combo_aggressive PIPE/local_opt_iters=5 PIPE/local_opt_interval=10 PIPE/keyframe_select_max=3 MODEL/sh_degree=0 PIPE/TSDF/voxel_size=0.016 PIPE/TSDF/trunc_dist=0.064
run_exp exp_12_combo_balanced   PIPE/local_opt_iters=15 PIPE/local_opt_interval=8 PIPE/keyframe_select_max=5 MODEL/sh_degree=1 PIPE/TSDF/voxel_size=0.012 PIPE/TSDF/trunc_dist=0.048

echo ""
echo "========================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "Summary: $SUMMARY"
echo "========================================"
cat "$SUMMARY"
