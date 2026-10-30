#!/usr/bin/env bash
# Evaluate a SmolVLA checkpoint on one tri_yolo seed, inside the gazebo_to_lerobot container.
#
#   eval_tri_seed.sh <checkpoint/pretrained_model> <seed> <out dir> [eval_smolvla_tri_sort.py args...]
#   PREPLACE=1 eval_tri_seed.sh ... <seed 1001-1400> ...   # the recorded target only, as recorded
#   HEADLESS=false eval_tri_seed.sh ...                     # Gazebo window, left open after the run
#   STEP_LOCKED=1 eval_tri_seed.sh ...                      # step_lock.py: reproducible to the bit
#   STEP_LOCKED=1 RESET_FROM=<run dir> eval_tri_seed.sh ... [--only <piece>]   # recompute from its snapshots
#   VIEWER=1 eval_tri_seed.sh ...                           # headless server + a view-only window
#   RECORD=1 eval_tri_seed.sh ...                           # Gazebo state log in <out>/gzlog*
#
# VIEWER=1 is how to watch a run that must match a headless one: the cameras and the physics are
# the headless server's, the window (gz sim -g) only displays them. HEADLESS=false renders the
# cameras through the window's own process, which can change their pixels. RECORD=1 keeps the
# simulation state every 20 ms of sim time; play_eval_gui.sh replays it without recomputing.
#
# Step-locked, every piece runs in a freshly started Gazebo server from its own start snapshot,
# in a batch exactly as in a repeat (30/10, test A: one Gazebo process carried something across
# pieces that the reset did not, and the yellow box parted from its batch run at step 52). A batch:
#   server 0          prepares piece 1 (observation pose, random start, gripper) -> snapshot_<1>.json
#   server k (1..n)   runs piece k from snapshot_<k>.json, then prepares piece k+1 -> snapshot_<k+1>.json
# A repeat (RESET_FROM) runs each piece asked for in a server of its own from the original's
# snapshot. A piece stopped by a step-locked check (a read-back, a stamp) is restarted from its
# snapshot in a new server, up to 3 times, each restart logged in restarts.log; the restarted run
# must reproduce every step the stopped one made, otherwise the seed stops with FATAL.
#
# Same scene as record_tri_seed.sh (tri_yolo.launch.py, piece_reach 0.28, headless, no YOLO),
# without the recorder. <out dir>/ gets result.json, trace_*.jsonl, snapshot_*.json and the logs.
# Everything is stopped on exit.
set -e
CKPT="$1"; SEED="$2"; OUT="$3"; shift 3 || true
[ -n "$OUT" ] || { echo "usage: $0 <checkpoint> <seed> <out dir> [args...]"; exit 2; }
[ -e "$OUT/result.json" ] && { echo "$OUT/result.json exists -- never overwrite a result"; exit 2; }
mkdir -p "$OUT"
HERE="$(dirname "$(readlink -f "$0")")"

set +u
source /opt/ros/jazzy/setup.bash
source /workspace/install/setup.bash
export DISPLAY="${TRI_DISPLAY:-:0}" LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe MESA_LOADER_DRIVER_OVERRIDE=
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
ros2 daemon stop >/dev/null 2>&1 || true
source "$HERE/tri_cleanup.sh"
stage() { echo "[$(TZ=Europe/Paris date +%H:%M:%S)] $*" | tee -a "$OUT/stages.log"; }
EVAL=(/workspace/venv_lerobot/bin/python -W ignore "$HERE/eval_smolvla_tri_sort.py" --checkpoint "$CKPT" --seed "$SEED" --out "$OUT")

scene_args() {  # <label>
    SCENE=(headless:="${HEADLESS:-true}")
    [ "${VIEWER:-0}" = 1 ] && SCENE=(headless:=true gui_client:=true)
    [ "${RECORD:-0}" = 1 ] && SCENE+=("gz_extra_args:=--record-path $OUT/gzlog$1 --record-period 0.02")
    return 0
}

wait_controllers() {
    # "inactive" contains "active": match the state as a whole word.
    until ros2 control list_controllers 2>/dev/null \
            | grep -qE 'gripper_position_controller.*[[:space:]]active([[:space:]]|$)'; do
        sleep 5
    done
    stage "controllers active"
}

# PREPLACE=1: replay a one-episode-per-layout recording (record_tri_seed.sh PREPLACE=1 RANDOM_START=1).
# preplace_pieces.py is deterministic in the seed, so it puts the same 0-3 pieces in their bins and
# names the same target; the placement is checked against the recording's preplaced.json
# (PREPLACE_REF, <dir>/seed_<seed>/preplaced.json), and only the target is evaluated, from the
# random start the recording drew first (draw 0 of default_rng(seed), --start-pose random).
preplace() {
    TARGET=$(python3 "$HERE/preplace_pieces.py" --seed "$SEED" --out "$OUT" 2>> "$OUT/preplace.log" | tail -1) \
        || { stage "FATAL: preplace failed: $(tail -1 "$OUT/preplace.log")"; exit 1; }
    python3 - "$OUT/preplaced.json" "${PREPLACE_REF:-/workspace/tri_sort_B}/seed_$SEED/preplaced.json" \
            >> "$OUT/stages.log" <<'EOF' || { stage "FATAL: preplacement differs from the recording"; exit 1; }
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
d = max(max(abs(x - y) for x, y in zip(a['pieces_after'][k], b['pieces_after'][k])) for k in b['pieces_after'])
ok = a['target'] == b['target'] and sorted(a['presorted']) == sorted(b['presorted']) and d < 0.002
print(f"preplacement vs recording: target {a['target']}, presorted {sorted(a['presorted'])}, max diff {d * 1000:.1f} mm, {'OK' if ok else 'DIFFERENT'}")
sys.exit(0 if ok else 1)
EOF
    stage "preplaced like the recording; target $TARGET"
}

if [ "${STEP_LOCKED:-0}" != 1 ]; then
    scene_args ""
    ros2 launch mycobot_gateway tri_yolo.launch.py seed:="$SEED" piece_reach:=0.28 \
        "${SCENE[@]}" yolo:=false > "$OUT/scene.log" 2>&1 &
    stage "seed $SEED: scene starting ($CKPT)"
    wait_controllers
    EXTRA=()
    if [ "${PREPLACE:-0}" = 1 ]; then
        preplace
        EXTRA+=(--only "$TARGET" --start-pose random)
    fi
    "${EVAL[@]}" "${EXTRA[@]}" "$@" > "$OUT/eval.log" 2>&1 || true
else
    # The server starts paused with the state_reset system, no wall-timed observation move; the
    # policy has the same thread count in every run (batch or window), the server one thread for
    # everything OpenMP (physics does not use it; the renderer's image libraries do).
    export GZ_SIM_SYSTEM_PLUGIN_PATH="$HERE/gz_systems${GZ_SIM_SYSTEM_PLUGIN_PATH:+:$GZ_SIM_SYSTEM_PLUGIN_PATH}"
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 STEP_LOCK_DIR="$OUT"
    # --only <pieces> chooses the pieces; every other argument goes to each evaluator call.
    ONLY=""; ARGS=()
    while [ $# -gt 0 ]; do
        if [ "$1" = --only ]; then ONLY="$2"; shift 2; else ARGS+=("$1"); shift; fi
    done
    START=()
    [ "${PREPLACE:-0}" = 1 ] && START=(--start-pose random)

    # One server: start it paused, the robot spawned at simulation time 0 (controller loop and
    # cameras on one time grid, step_lock.py); then the simulation only advances in 100-step blocks
    # until the evaluator locks it (it touches lock_requested, this stepper answers stepper_stopped).
    # multi_step requests add up: the next block is sent only once /stats shows the previous one
    # done, so nothing is left queued when the evaluator locks. A request can time out unanswered
    # while the policy loads (29/10): if the iterations have not moved after 60 s, the block was not
    # queued and is sent again; the last block always finishes before stepper_stopped. A /stats
    # message without an iterations field means 0 (protobuf text leaves zeros out); no message at
    # all prints nothing, "unknown" (29/10: compared as 0, the wait never ended).
    iterations() { gz topic -e -n 1 -t /world/tri_yolo/stats 2>/dev/null \
        | awk '/^iterations:/ {n = $2} /sim_time|real_time|paused/ {got = 1} END {if (got) print n + 0}'; }
    server_up() {  # <label>
        rm -f "$OUT/lock_requested" "$OUT/stepper_stopped"
        scene_args "_$1"
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ros2 launch mycobot_gateway tri_yolo.launch.py seed:="$SEED" piece_reach:=0.28 \
            "${SCENE[@]}" run_on_start:=false state_reset:=true observe:=false yolo:=false > "$OUT/scene_$1.log" 2>&1 &
        stage "seed $SEED, server $1: scene starting ($CKPT)"
        until gz model --list 2>/dev/null | grep -q mycobot_320; do sleep 1; done
        stage "robot spawned at sim time 0; stepping in 100-step blocks until lock"
        ( while [ ! -e "$OUT/lock_requested" ]; do
              start=$(iterations)
              if [ -z "$start" ]; then sleep 0.5; continue; fi
              gz service -s /world/tri_yolo/control --reqtype gz.msgs.WorldControl --reptype gz.msgs.Boolean \
                  --timeout 5000 --req 'multi_step: 100, pause: true' > /dev/null || true
              t0=$SECONDS
              while :; do
                  now=$(iterations)
                  if [ -n "$now" ] && [ "$now" -ge $((start + 100)) ]; then break; fi
                  if [ $((SECONDS - t0)) -ge 60 ]; then break; fi
                  sleep 0.2
              done
          done
          prev=x; t0=$SECONDS
          while now=$(iterations); { [ -z "$now" ] || [ "$now" != "$prev" ]; } && [ $((SECONDS - t0)) -lt 60 ]; do
              prev=$now; sleep 2
          done
          touch "$OUT/stepper_stopped" ) &
        wait_controllers
    }
    server_down() {
        tri_cleanup
        set -e
        ros2 daemon stop >/dev/null 2>&1 || true
    }
    # Run piece <name> (index <k>) from its snapshot in fresh servers until it ends without a
    # step-locked stop (at most 3 restarts); then the extra arguments, e.g. --prepare-next.
    run_piece() {  # <name> <k> [args...]
        local name=$1 k=$2 attempt=1; shift 2
        while :; do
            server_up "${name}_$attempt"
            "${EVAL[@]}" --step-locked --only "$name" --reset-from "$FROM" --attempt $attempt \
                --result-name "result_$name.json" "${START[@]}" "${ARGS[@]}" "$@" > "$OUT/eval_${name}_$attempt.log" 2>&1 || true
            stop=$(python3 -c "import json,sys; e=json.load(open(sys.argv[1]))['pieces'][0].get('error', ''); \
print(e if e.startswith('step-locked:') else '')" "$OUT/result_$name.json" 2>/dev/null || echo "no result")
            # the last piece's viewer window stays open after the run, until it is closed
            if [ -z "$stop" ] && [ "$name" = "$LAST" ] && [ "${VIEWER:-0}" = 1 ] && [ "${KEEP_GUI:-1}" = 1 ]; then
                LEFT_UP=1; break
            fi
            server_down
            [ -z "$stop" ] && break
            if [ $attempt -gt "${RESTARTS:-3}" ]; then stage "FATAL: $name stopped after ${RESTARTS:-3} restarts: $stop"; exit 1; fi
            echo "[$(TZ=Europe/Paris date +%H:%M:%S)] $name: restart $attempt after $stop" | tee -a "$OUT/restarts.log"
            cp "$OUT/result_$name.json" "$OUT/result_$name.attempt$attempt.json" 2>/dev/null || true
            attempt=$((attempt + 1))
        done
        # a restarted piece must reproduce every step its stopped attempts made
        for f in "$OUT"/trace_"$name".stopped_*.jsonl; do
            [ -e "$f" ] || continue
            python3 - "$OUT/trace_$name.jsonl" "$f" <<'EOF' || { stage "FATAL: the restart of $name does not reproduce $(basename "$f")"; exit 1; }
import json, sys
done = [json.loads(l) for l in open(sys.argv[1]).read().splitlines()[1:]]
strip = lambda s: {k: v for k, v in s.items() if k != 'commands'}   # diagnostics, differ by design
done = [strip(s) for s in done if 'action' in s]
stopped = [strip(json.loads(l)) for l in open(sys.argv[2]).read().splitlines()]
same = done[:len(stopped)] == stopped
print(f"{sys.argv[2]}: {len(stopped)} steps, {'reproduced' if same else 'NOT reproduced'} by the restart")
sys.exit(0 if same else 1)
EOF
        done
        stage "$name done: $(python3 -c "import json,sys; p=json.load(open(sys.argv[1]))['pieces'][0]; tr=p.get('tracking', {}); \
print(p.get('error') or ('sorted' if p.get('placed_after_pick') else 'picked' if p.get('picked') else 'not picked'), \
'| tracking median', tr.get('median_deg'), 'deg, fast periods', tr.get('fast_periods'), \
'ACCEPTED' if tr.get('accepted') else 'NOT ACCEPTED')" "$OUT/result_$name.json")"
    }

    ALL=(cube_rouge pave_jaune cylindre_vert cube_bleu)          # TRI_TARGETS order
    LEFT_UP=0
    if [ -n "${RESET_FROM:-}" ]; then
        FROM="$RESET_FROM"
        if [ -n "$ONLY" ]; then PIECES=(${ONLY//,/ }); else
            PIECES=(); for p in "${ALL[@]}"; do
                { [ -e "$FROM/snapshot_$p.json" ] || [ -e "$FROM/trace_$p.jsonl" ]; } && PIECES+=("$p"); done
        fi
        LAST="${PIECES[-1]}"
        for k in "${!PIECES[@]}"; do run_piece "${PIECES[$k]}" "$k"; done
    else
        FROM="$OUT"
        server_up prepare
        if [ "${PREPLACE:-0}" = 1 ]; then preplace; PIECES=("$TARGET"); fi
        [ "${PREPLACE:-0}" = 1 ] || { if [ -n "$ONLY" ]; then PIECES=(${ONLY//,/ }); else PIECES=("${ALL[@]}"); fi; }
        "${EVAL[@]}" --step-locked --prepare "${PIECES[0]}" --start-index 0 "${START[@]}" "${ARGS[@]}" \
            > "$OUT/eval_prepare.log" 2>&1 || true
        server_down
        [ -e "$OUT/snapshot_${PIECES[0]}.json" ] || { stage "FATAL: no start snapshot for ${PIECES[0]}"; exit 1; }
        LAST="${PIECES[-1]}"
        for k in "${!PIECES[@]}"; do
            NEXT=()
            n=$((k + 1))
            [ $n -lt ${#PIECES[@]} ] && NEXT=(--prepare-next "${PIECES[$n]}" --start-index $n)
            run_piece "${PIECES[$k]}" "$k" "${NEXT[@]}"
            [ ${#NEXT[@]} -eq 0 ] || [ -e "$OUT/snapshot_${PIECES[$n]}.json" ] \
                || { stage "FATAL: no start snapshot for ${PIECES[$n]}"; exit 1; }
        done
    fi
    # result.json: the pieces in order, as the single-server evaluator wrote it
    python3 - "$OUT" "${PIECES[@]}" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
parts = [json.loads((out / f'result_{p}.json').read_text()) for p in sys.argv[2:]]
res = dict(parts[0], pieces=[x['pieces'][0] for x in parts], per_piece_servers=True)
(out / 'result.json').write_text(json.dumps(res, indent=1) + '\n')
EOF
fi
stage "done: $(python3 -c "import json,sys; r=json.load(open(sys.argv[1]))['pieces']; \
print(sum(p.get('placed_after_pick', False) for p in r), '/', len(r), 'sorted')" "$OUT/result.json" 2>/dev/null)"

# With the GUI (HEADLESS=false), the scene stays up after the evaluation, simulation running, until
# the Gazebo window is closed or Ctrl+C: any Gazebo process ending (the window) or the launch
# ending stops everything (tri_cleanup). KEEP_GUI=0 closes it at once, as headless runs do.
# Step-locked, each piece's server (and its window) closes when the piece is done, but the last.
if { [ "${STEP_LOCKED:-0}" != 1 ] && { [ "${HEADLESS:-true}" = false ] || [ "${VIEWER:-0}" = 1 ]; } \
        && [ "${KEEP_GUI:-1}" = 1 ]; } || [ "${LEFT_UP:-0}" = 1 ]; then
    stage "Gazebo window left open: close it (or Ctrl+C here) to stop the scene"
    n=$(pgrep -fc "gz sim" || true)
    while kill -0 %1 2>/dev/null && [ "$(pgrep -fc "gz sim" || true)" -ge "$n" ]; do sleep 2; done
    stage "Gazebo window closed: stopping the scene"
fi
