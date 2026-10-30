#!/usr/bin/env python3
"""Two evaluations of the same seed, piece by piece: identical, or where they first part.

  compare_runs.py <run dir A> <run dir B> [--json out.json]

Step-locked runs (eval_smolvla_tri_sort.py --step-locked) carry a fingerprint per piece: equal
fingerprints mean the same start state to the bit, the same state and action at every step, the
same camera frames and the same piece poses. Otherwise, and for free-running runs, the traces are
compared: the first step where the state or the action differs (and by how much, in degrees), the
first camera frame that differs, and the outcome of each run. Pieces B did not run (a one-piece
recompute) are listed and left out. Exit 0 when every piece B ran is identical. Plain python3 with numpy.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load(run):
    res = json.loads((run / 'result.json').read_text())
    pieces = {}
    for p in res['pieces']:
        f = run / f"trace_{p['piece']}.jsonl"
        lines = [json.loads(line) for line in f.read_text().splitlines()] if f.exists() else [{}]
        pieces[p['piece']] = {'result': p, 'head': lines[0], 'steps': [x for x in lines[1:] if 'action' in x],
                              'frames': [x['frame'] for x in lines[1:] if 'frame' in x]}
    return res, pieces


def outcome(r):
    if 'error' in r:
        return f"error: {r['error']}"
    return ('sorted' if r.get('placed_after_pick') else 'picked, not placed' if r.get('picked')
            else 'not picked') + f", {r.get('steps')} steps"


def quantities(full):
    """Flatten a Gazebo state (step_lock.gz_state(pieces=True)) into {quantity name: value}."""
    out = {}
    for kind in ('positions', 'velocities'):
        for joint, v in full.get(kind, {}).items():
            out[f'joint {joint} {kind[:-1]}'] = [v]
    for piece, s in (full.get('pieces') or {}).items():
        for kind, v in s.items():
            out[f'{piece} {kind}'] = v
    return out


def first_quantity(x, y):
    """The quantity that differs most between two Gazebo states, and by how much; None if equal."""
    qx, qy = quantities(x), quantities(y)
    worst = None
    for key in sorted(set(qx) | set(qy)):
        if key not in qx or key not in qy:
            return f'{key} present in one run only'
        d = float(np.abs(np.array(qx[key]) - np.array(qy[key])).max())
        if d and (worst is None or d > worst[1]):
            worst = (key, d)
    return None if worst is None else f'{worst[0]} by {worst[1]:.3g}'


def first_full_difference(a, b):
    """Step-locked runs with the full Gazebo state per step: where the physical state first parts."""
    ha, hb = a['head'].get('after_reset'), b['head'].get('after_reset')
    if ha is not None and hb is not None:
        q = first_quantity(ha, hb)
        if q:
            return f'state after the exact reset already differs: {q}'
    for k, (x, y) in enumerate(zip(a['steps'], b['steps'])):
        if 'full' not in x or 'full' not in y:
            return None
        q = first_quantity(x['full'], y['full'])
        if q:
            return f'Gazebo state first differs at step {k} (chunk {x["chunk"]}): {q}'
    return 'Gazebo state equal at every step'


def first_difference(a, b):
    sa, sb = a['head'].get('start_state'), b['head'].get('start_state')
    if sa is not None and sb is not None and sa != sb:
        d = np.abs(np.array(sa) - np.array(sb))
        note = f'start state differs: joints up to {np.degrees(d[:6]).max():.3g} deg, gripper {d[6]:.3g}'
    else:
        note = 'same start state' if sa is not None else 'no full-precision start state (free-running run)'
    for k, (x, y) in enumerate(zip(a['steps'], b['steps'])):
        if x['state'] != y['state'] or x['action'] != y['action']:
            ds = np.degrees(np.abs(np.array(x['state'][:6]) - np.array(y['state'][:6]))).max()
            da = np.degrees(np.abs(np.array(x['action'][:6]) - np.array(y['action'][:6]))).max()
            note += f'; first difference at step {k} (chunk {x["chunk"]}): state {ds:.3g} deg, action {da:.3g} deg'
            break
    else:
        if len(a['steps']) != len(b['steps']):
            note += f'; same steps up to {min(len(a["steps"]), len(b["steps"]))}, then one run stops'
        else:
            note += '; every step equal'
    full = first_full_difference(a, b)
    if full:
        note += f'; {full}'
    for k, (x, y) in enumerate(zip(a['frames'], b['frames'])):
        if x['sha256'] != y['sha256']:
            cams = [c for c in x['sha256'] if x['sha256'][c] != y['sha256'].get(c)]
            note += f'; first different camera frame at step {x["step"]} ({", ".join(cams)})'
            break
    return note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a', type=Path)
    ap.add_argument('b', type=Path)
    ap.add_argument('--json', type=Path)
    args = ap.parse_args()
    (ra, pa), (rb, pb) = load(args.a), load(args.b)
    for key in ('checkpoint', 'seed', 'start_pose', 'n_action_steps', 'max_steps'):
        if ra.get(key) != rb.get(key):
            print(f'NOT THE SAME RUN: {key} {ra.get(key)} vs {rb.get(key)}')
    rows, all_same = [], True
    for piece in pa:
        a, b = pa[piece], pb.get(piece)
        if b is None:
            rows.append({'piece': piece, 'identical': None, 'a': outcome(a['result']), 'b': 'not run'})
            continue
        fa, fb = a['result'].get('fingerprint'), b['result'].get('fingerprint')
        same = fa is not None and fa == fb
        all_same &= same
        rows.append({'piece': piece, 'identical': same, 'a': outcome(a['result']), 'b': outcome(b['result']),
                     'same_outcome': outcome(a['result']) == outcome(b['result']),
                     'detail': 'identical fingerprints' if same else first_difference(a, b)})
    print(f"{'piece':15s}{args.a.name:>32s}{args.b.name:>32s}   identical")
    for r in rows:
        print(f"{r['piece']:15s}{r['a']:>32s}{r['b']:>32s}   "
              f"{'-' if r['identical'] is None else 'YES' if r['identical'] else 'no'}")
        if r.get('detail'):
            print(f"{'':15s}{r['detail']}")
    ran = [r for r in rows if r['identical'] is not None]
    all_same = all_same and bool(ran)
    print(f'ALL {len(ran)} PIECES RUN IDENTICAL' if all_same else 'NOT IDENTICAL (see above)')
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1) + '\n')
    sys.exit(0 if all_same else 1)


if __name__ == '__main__':
    main()
