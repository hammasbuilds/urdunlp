"""Fetch the Urdu and Sindhi parts of Google's Dakshina dataset.

    python scripts/fetch_dakshina.py [--out data/dakshina]

Dakshina (Roark et al., LREC 2020, CC BY-SA 4.0) is the only public source of Roman
Urdu that people actually typed and that is paired with the Urdu it stands for:

  * `ur/lexicons/*.tsv` - native word, a romanisation, and how many of the
    annotators wrote that romanisation. One Urdu word has several attested
    spellings, which is exactly what `roman_key` has to collapse.
  * `ur/romanized/ur.romanized.rejoined.tsv` - 10,000 Wikipedia sentences, each
    romanised by hand, so transliteration is scored against sentences rather
    than isolated words.
  * `sd/native_script_wikipedia/...valid.text.shuf.txt.gz` - Sindhi prose, a
    Perso-Arabic language that `is_urdu` used to accept as Urdu.

The whole archive is one 2 GB uncompressed tar. Nothing here needs more than 10 MB
of it, and the connection this was built on ran at ~300 KB/s, so rather than
download the tar, this walks its 512-byte headers with HTTP range requests and
fetches only the members it needs. The member offsets are read from the headers
every time, not hard-coded, so a republished archive still works.

Standard library only. The fetched data is not part of the package and is not
needed by the test suite.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"
ROOT = "dakshina_dataset_v1.0/"

WANTED = {
    "ur/lexicons/ur.translit.sampled.train.tsv",
    "ur/lexicons/ur.translit.sampled.dev.tsv",
    "ur/lexicons/ur.translit.sampled.test.tsv",
    "ur/romanized/ur.romanized.rejoined.tsv",
    "ur/romanized/ur.romanized.rejoined.aligned.tsv",
    # Dakshina's own dev/test split of the romanised sentences. Scoring on the
    # published split, rather than one invented here, keeps the numbers comparable.
    "ur/romanized/ur.romanized.rejoined.dev.native.txt",
    "ur/romanized/ur.romanized.rejoined.test.native.txt",
    "sd/native_script_wikipedia/sd.wiki-filt.valid.text.shuf.txt.gz",
    # Dakshina's *training* partition of Urdu Wikipedia, which the transliteration
    # vocabulary is counted from. The romanised evaluation sentences were drawn from
    # the held-out partition, so counting words here cannot see the test sentences.
    # measure_translit.py checks that claim rather than trusting it.
    "ur/native_script_wikipedia/ur.wiki-filt.train.text.shuf.txt.gz",
}

# Urdu sorts last in the archive. Starting the header walk at the Sindhi directory
# skips ~1,200 header requests that would each cost a round trip; if the archive is
# ever republished with a different layout the walk falls back to offset 0.
HINT_OFFSET = 1197286400 - 512


def fetch_range(start: int, end: int) -> bytes:
    request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def walk(start: int):
    """Yield (name, size, data_offset) for each regular member from `start`."""
    offset = start
    long_name = None
    while True:
        header = fetch_range(offset, offset + 511)
        if len(header) < 512 or header == b"\0" * 512:
            return
        name = header[0:100].rstrip(b"\0").decode("utf-8", "replace")
        prefix = header[345:500].rstrip(b"\0").decode("utf-8", "replace")
        size = int(header[124:136].rstrip(b"\0 ").decode() or "0", 8)
        kind = header[156:157]
        data = offset + 512
        if kind == b"L":  # GNU long name: the next header's name is in this body
            long_name = fetch_range(data, data + size - 1).rstrip(b"\0").decode()
        else:
            full = long_name or (f"{prefix}/{name}" if prefix else name)
            long_name = None
            if kind in (b"0", b"\0"):
                yield full, size, data
        offset = data + ((size + 511) // 512) * 512


def valid_header_at(offset: int) -> bool:
    header = fetch_range(offset, offset + 511)
    checksum = int(header[148:156].rstrip(b"\0 ").decode() or "-1", 8)
    return checksum == sum(header[:148]) + 8 * 32 + sum(header[156:])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data/dakshina"))
    args = ap.parse_args()

    start = HINT_OFFSET if valid_header_at(HINT_OFFSET) else 0
    remaining = set(WANTED)
    for full, size, data in walk(start):
        relative = full.removeprefix(ROOT)
        if relative not in remaining:
            continue
        target = args.out / relative
        if target.exists() and target.stat().st_size == size:
            print(f"have  {relative}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"fetch {relative} ({size:,} bytes)", flush=True)
            target.write_bytes(fetch_range(data, data + size - 1))
        remaining.discard(relative)
        if not remaining:
            break

    if remaining:
        print("not found in the archive:", *sorted(remaining), sep="\n  ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
