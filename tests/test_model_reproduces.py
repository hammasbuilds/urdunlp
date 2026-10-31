"""The shipped language-ID model rebuilds bit-identically from the committed corpus.

A model in git drifts from the code that made it. `FINGERPRINT.tsv` records the corpus
the shipped model was built from, but nothing checked that rebuilding from that corpus
*reproduces* it - and a sibling repository shipped two stale result artefacts for three
days for exactly that reason, with the test comparing one stale file against another.

This retrains from `data/wiki_large` at the documented hyperparameters and compares the
decompressed payload's SHA-256 against the model in the package. The gzip framing is not
compared, because it stores a timestamp.

Slow - it rebuilds the model - so it is marked and deselected by default. Run it with
`pytest -m slow tests/test_model_reproduces.py`.
"""

from __future__ import annotations

import gzip
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = ROOT / "src" / "urdunlp" / "data" / "langid.json.gz"
CORPUS = ROOT / "data" / "wiki_large"
BUILDER = ROOT / "scripts" / "build_langid_models.py"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (SHIPPED.exists() and CORPUS.is_dir() and BUILDER.exists()),
        reason="the training corpus is not shipped with the tests",
    ),
]


def payload_digest(path: Path) -> tuple[str, int]:
    raw = gzip.decompress(path.read_bytes())
    return hashlib.sha256(raw).hexdigest(), len(raw)


def test_retraining_reproduces_the_shipped_model(tmp_path):
    out = tmp_path / "model"
    out.mkdir()
    result = subprocess.run(
        [sys.executable, str(BUILDER),
         "--wiki-dir", str(CORPUS), "--out-dir", str(out), "--skip-tagger"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=3600,
    )
    assert result.returncode == 0, (
        f"the builder failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
    )
    rebuilt = out / "langid.json.gz"
    assert rebuilt.is_file(), (
        f"the builder exited 0 but wrote no model to {out}. "
        f"stdout:\n{result.stdout[-2000:]}"
    )

    got, got_n = payload_digest(rebuilt)
    want, want_n = payload_digest(SHIPPED)
    assert (got, got_n) == (want, want_n), (
        f"rebuilding from {CORPUS.name} gives a payload of {got_n:,} bytes "
        f"({got[:16]}), but the shipped model is {want_n:,} bytes ({want[:16]}). "
        f"Either the corpus, the hyperparameters or the builder has changed since the "
        f"shipped model was made, and FINGERPRINT.tsv no longer identifies it."
    )
