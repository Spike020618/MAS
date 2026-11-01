"""Read-only local experiment checkpoint inventory. Never calls model APIs."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = ('native_heterogeneous_new64_v1', 'native_reviewer_calibration_v1')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_run(root, workspace=ROOT):
    root = Path(root)
    plan_path = root / 'plan.json'
    if not plan_path.exists():
        return {'run': root.name, 'state': 'missing_plan', 'automatic_resume_ready': False}
    plan = json.loads(plan_path.read_text())
    drift = []
    snapshots = []
    for name, expected in plan.get('source_sha256', {}).items():
        live = Path(workspace) / 'experiments/paired_deliberation' / name
        snap = root / 'source_snapshot' / name
        if not live.exists() or digest(live) != expected:
            drift.append(name)
        if snap.exists():
            snapshots.append({'file': name, 'matches_plan': digest(snap) == expected})
    # An intent alone is ambiguous even when a parsed cache is present: retain it.
    providers = []
    slot_dirs = [root/'slots'] + [p for p in root.glob('*/slots') if p.is_dir()]
    for slots in slot_dirs:
        if not slots.is_dir():
            continue
        raw = list(slots.glob('*.raw.json'))
        unknown = [p.name for p in slots.glob('*.intent.json')
                   if not (slots/(p.name.removesuffix('.intent.json')+'.raw.json')).exists()]
        providers.append({'scope': str(slots.relative_to(root)), 'raw_receipts': len(raw),
                          'unknown_requests_do_not_resend': sorted(unknown)})
    sealed = (root/'generation_complete.json').exists()
    scored = (root/'summary.json').exists()
    return {'run': root.name, 'plan_sha256': digest(plan_path),
            'state': 'scored' if sealed and scored else 'sealed_unscored' if sealed else 'incomplete',
            'completed_cases': len(list((root/'cases').glob('case_*.json'))),
            'providers': providers, 'source_drift': drift, 'snapshots': snapshots,
            'automatic_resume_ready': not sealed and not drift and not any(
                p['unknown_requests_do_not_resend'] for p in providers),
            'resume_note': 'Completed runs need no generation. Source drift requires a new version; unknown requests require manual reconciliation, never blind resend.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='append', help='Result directory; may repeat')
    args = p.parse_args()
    roots = [Path(x) for x in args.run] if args.run else [ROOT/'experiments/results'/x for x in DEFAULTS]
    print(json.dumps({'mode': 'read_only_no_network_no_secrets', 'runs': [inspect_run(x) for x in roots]}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
