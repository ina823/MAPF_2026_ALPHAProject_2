"""Build registry/il_input_reference_v1.json from the Alpha 1 training npz. Read-only.

Opens real_v03_fd.npz with allow_pickle=False and checkpoints/MANIFEST.json,
never writes next to them, and creates the reference file only if it does not
exist (never overwrites). Deterministic: the same npz gives the same bytes.

Exit codes: 0 written, 1 npz/manifest mismatch, 2 refused (output exists / npz missing).

Usage (from the repository root):
    python -m scripts.build_il_input_reference [--npz ../Alpha1_legacy/real_v03_fd.npz] [--out registry/il_input_reference_v1.json]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.il_compat import build_reference, reference_json
from src.data.registry import create_files_exclusive

DEFAULT_NPZ = PROJECT_ROOT.parent / "Alpha1_legacy" / "real_v03_fd.npz"
DEFAULT_OUT = PROJECT_ROOT / "registry" / "il_input_reference_v1.json"
MANIFEST = PROJECT_ROOT / "checkpoints" / "MANIFEST.json"
MAX_NORMALIZATION_DIFF = 1e-5   # float32 training stats vs float64 recomputation


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", default=str(DEFAULT_NPZ))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)
    npz, out = Path(args.npz), Path(args.out)
    if out.exists():
        print(f"REFUSED: {out} already exists", file=sys.stderr)
        return 2
    if not npz.is_file():
        print(f"REFUSED: npz not found: {npz}", file=sys.stderr)
        return 2
    ref = build_reference(npz, MANIFEST)
    diff = ref["normalization_check"]["max_abs_diff"]
    if diff > MAX_NORMALIZATION_DIFF:
        print(f"FAILED: npz statistics differ from the checkpoint normalization by {diff}", file=sys.stderr)
        return 1
    create_files_exclusive([(out, reference_json(ref))])
    print(f"npz sha256={ref['source']['sha256']} n={ref['source']['n_samples']} "
          f"normalization max_abs_diff={diff:.2e}")
    for name, f in ref["features"].items():
        print(f"  {name:17s} p0.5={f['p0.5']:8.3f} p99.5={f['p99.5']:8.3f} min={f['min']:6.1f} max={f['max']:6.1f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
