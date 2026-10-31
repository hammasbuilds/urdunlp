"""The sweep's selection rule, which decides what the wheel ships.

The rule is "the smallest model among the settings validation cannot separate", and
"cannot separate" was a constant: `MEAN_TOLERANCE = 0.002` on the mean accuracy over four
window lengths. The paragraph justifying that number argues from the standard error of a
single rate - 0.95 points for Saraiki on 1,827 validation paragraphs - which is the right
error bar for "how well do we know this model's accuracy" and the wrong one for "is A
better than B", because every setting is scored on the same paragraphs.

It mattered. The rule picked `top=120000` over `top=180000` on a val_mean gap of 0.0020,
just inside the tolerance. Paragraph by paragraph at 20 characters, 180000 is right where
120000 is wrong 149 times and wrong where it is right 98 times: an exact two-sided p of
0.0014. The ladder had not flattened, and a tolerance expressed in unpaired units could
not see that it had not.

So these tests hold the rule to both directions - it must choose the larger model when the
difference is real, and the smaller one when it is not - and to the arithmetic of the test
itself, with the real counts as one of the cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sweep_langid import WINDOWS, choose, mcnemar, separable  # noqa: E402


def pair(a_only: int, b_only: int, both: int, neither: int) -> tuple[bytes, bytes]:
    """Two outcome vectors with the given 2x2 table, in a fixed order."""
    a = [1] * a_only + [0] * b_only + [1] * both + [0] * neither
    b = [0] * a_only + [1] * b_only + [1] * both + [0] * neither
    return bytes(a), bytes(b)


def setting(top: int, mean: float, size: int, outcomes: bytes, skr: float = 0.8) -> dict:
    return {
        "n_max": 6,
        "alpha": 0.03,
        "top": top,
        "val_mean": mean,
        "val_skr_20": skr,
        "json_bytes": size,
        "outcomes": {w: outcomes for w in WINDOWS},
    }


def test_the_real_discordant_counts_give_the_measured_p_value() -> None:
    """The case the rule was wrong on, with the counts from the validation run."""
    a, b = pair(98, 149, 18778, 822)
    wins, losses, p = mcnemar(a, b)
    assert (wins, losses) == (98, 149)
    assert p == pytest.approx(0.0014, abs=5e-5), (
        "the 20-character comparison between top=120000 and top=180000 must come out "
        "separable; it is the comparison the old tolerance band could not make"
    )


def test_no_disagreement_is_not_a_difference() -> None:
    a, b = pair(0, 0, 100, 10)
    assert mcnemar(a, b) == (0, 0, 1.0)


def test_a_symmetric_split_is_not_a_difference() -> None:
    a, b = pair(20, 20, 19627, 180)
    wins, losses, p = mcnemar(a, b)
    assert (wins, losses) == (20, 20)
    assert p == 1.0


def test_vectors_of_different_lengths_are_refused() -> None:
    """Silently comparing the first n would compare different paragraphs.

    Which paragraphs get scored depends on the window function alone, so two settings
    agree - but the day that stops being true, the rule must stop rather than choose a
    model on a misaligned comparison.
    """
    with pytest.raises(ValueError, match="different lengths"):
        mcnemar(b"\x01\x01\x01", b"\x01\x01")


def test_the_larger_model_wins_when_the_difference_is_real() -> None:
    a, b = pair(98, 149, 18778, 822)
    small = setting(120000, 0.95253, 43809165, a)
    big = setting(180000, 0.95450, 66300000, b, skr=0.8144)
    assert separable(small, big) is not None, "the ceiling is ahead and the rule must see it"
    assert choose([small, big])["top"] == 180000


def test_the_smaller_model_wins_when_it_is_not() -> None:
    """The rule's whole point: wheel size is paid by every user, so a tie goes to small."""
    _, b = pair(20, 20, 19627, 180)
    small = setting(60000, 0.95450, 20019085, b)
    big = setting(180000, 0.95450, 66300000, b)
    assert separable(small, big) is None
    assert choose([small, big])["top"] == 60000


def test_a_candidate_that_wins_a_length_is_not_excluded_for_it() -> None:
    """Only a loss counts against a candidate.

    The ceiling is the setting with the best mean. A smaller model that beats it at some
    window length is exactly what the sweep is looking for, so `separable` must test the
    ceiling being ahead and not merely the two differing.
    """
    a, b = pair(149, 98, 18778, 822)  # the candidate wins, not the ceiling
    candidate = setting(60000, 0.9500, 20019085, a)
    ceiling = setting(180000, 0.9545, 66300000, b)
    assert separable(candidate, ceiling) is None
    assert choose([candidate, ceiling])["top"] == 60000


def test_rows_without_outcomes_fall_back_to_the_tolerance_band() -> None:
    """A report reloaded from disk has no outcome vectors; it must still be choosable."""
    small = {k: v for k, v in setting(60000, 0.9530, 20019085, b"").items() if k != "outcomes"}
    big = {k: v for k, v in setting(180000, 0.9545, 66300000, b"").items() if k != "outcomes"}
    assert choose([small, big])["top"] == 60000
    far = {**small, "val_mean": 0.9400}
    assert choose([far, big])["top"] == 180000
