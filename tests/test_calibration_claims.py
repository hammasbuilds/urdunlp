"""Section 5.1's calibration numbers, and the cancellation that hides them.

The section's argument is not that the model is badly calibrated - it is that the *pooled*
figure says the opposite of every subgroup, because overconfidence on the languages the
model fails and underconfidence on the ones it gets right subtract from each other. That is
the kind of claim a later edit can quietly invert by averaging differently, so it is pinned
here against the artefact rather than left in prose.

Three properties carry the section and each has its own test: the pooled ECE is far smaller
than the balanced one; the balanced *signed* gap is far smaller than the balanced absolute
one, which is the cancellation; and the sign of the error is positive exactly on the
languages whose accuracy is worst. If any of them stopped holding the section would be
wrong, not merely stale.
"""

from __future__ import annotations

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MEASURED = ROOT / "research" / "results" / "calibration.json"
PAPER = ROOT / "paper" / "main.md"

HEADING = "## 5.1 The confidence does not notice"

# As section 5.1 publishes them.
PUBLISHED_TEMPERATURE = 1.8944
PUBLISHED_IN_DOMAIN_N = 6_600
PUBLISHED_IN_DOMAIN_ACC = 0.9229
PUBLISHED_IN_DOMAIN_ECE_RAW = 0.0314
PUBLISHED_IN_DOMAIN_ECE_SCALED = 0.0132
PUBLISHED_POOLED_N = 120_934
PUBLISHED_POOLED_ECE_RAW = 0.0044
PUBLISHED_POOLED_ECE_SCALED = 0.0244
PUBLISHED_BALANCED_ACC = 0.8921
PUBLISHED_BALANCED_ECE = 0.0461
PUBLISHED_BALANCED_SIGNED = -0.0066
PUBLISHED_PER_LANGUAGE = {
    "ur": (2_812, 0.5302, 0.0938, 0.0918),
    "pnb": (70, 0.6857, 0.1108, 0.0082),
    "skr": (133, 0.9474, 0.0817, -0.0714),
    "ar": (64_042, 0.9861, 0.0272, -0.0272),
    "fa": (31_437, 0.9895, 0.0456, -0.0456),
    "ckb": (12_491, 0.9982, 0.0022, -0.0017),
    "ug": (9_915, 0.9998, 0.0004, -0.0002),
    "ks": (34, 1.0000, 0.0069, -0.0069),
}


def _measured():
    if not MEASURED.exists():
        pytest.skip("research/results/calibration.json is not present")
    return json.loads(MEASURED.read_text(encoding="utf-8"))


def _repo_file(path: pathlib.Path) -> str:
    """A file that lives in the repository rather than in the wheel."""
    if not path.is_file():
        pytest.skip(f"{path.name} is not present - running against an installed wheel")
    return path.read_text(encoding="utf-8")


def _section() -> str:
    text = _repo_file(PAPER)
    start = text.index(HEADING)
    end = text.index("\n# 6. ", start)
    return " ".join(text[start:end].split())


# --------------------------------------------------------------------------- in domain


def test_the_temperature_was_fitted_on_a_split_the_paper_reports_nothing_else_from():
    fit = _measured()["fit_on_val"]
    assert fit["temperature"] == pytest.approx(PUBLISHED_TEMPERATURE, abs=5e-5)
    # A minimum at the bracket edge would be the search's answer, not the model's.
    assert not fit["at_the_bracket_edge"], fit
    # And it has to beat the uncalibrated likelihood, or there was nothing to fit.
    assert fit["nll"] < fit["nll_at_temperature_1"], fit


def test_scaling_halves_the_in_domain_error_without_changing_a_prediction():
    data = _measured()["in_domain"]
    raw, scaled = data["uncalibrated"], data["calibrated"]
    assert raw["n"] == scaled["n"] == PUBLISHED_IN_DOMAIN_N
    # Dividing every logit by one positive number preserves their order, so accuracy
    # must be identical. If it moved, the implementation is not temperature scaling.
    assert raw["accuracy"] == scaled["accuracy"] == pytest.approx(PUBLISHED_IN_DOMAIN_ACC, abs=5e-5)
    assert raw["ece"] == pytest.approx(PUBLISHED_IN_DOMAIN_ECE_RAW, abs=5e-5)
    assert scaled["ece"] == pytest.approx(PUBLISHED_IN_DOMAIN_ECE_SCALED, abs=5e-5)
    assert scaled["ece"] < raw["ece"] / 2.0
    # Overconfident before, marginally underconfident after.
    assert raw["overconfidence"] > 0 > scaled["overconfidence"]


# --------------------------------------------------- the pooled figure and why it lies


def test_the_pooled_out_of_domain_error_looks_excellent():
    """The number the section exists to discredit, pinned so it cannot quietly move."""
    pooled = _measured()["out_of_domain"]["pooled_uncalibrated"]
    assert pooled["n"] == PUBLISHED_POOLED_N
    assert pooled["ece"] == pytest.approx(PUBLISHED_POOLED_ECE_RAW, abs=5e-5)
    # It is an order of magnitude better than the in-domain figure, which is the
    # implausibility that prompted looking at the composition.
    assert pooled["ece"] < _measured()["in_domain"]["uncalibrated"]["ece"] / 5


def test_the_pool_is_dominated_by_the_languages_the_toolkit_is_least_about():
    per_language = _measured()["out_of_domain"]["per_language"]
    total = sum(row["n"] for row in per_language.values())
    easy = sum(per_language[code]["n"] for code in ("ar", "fa", "ckb", "ug"))
    assert easy / total > 0.97, (easy, total)
    # And the three the task is about are a rounding error of the mass.
    hard = sum(per_language[code]["n"] for code in ("ur", "pnb", "skr"))
    assert hard / total < 0.03, (hard, total)
    assert "97.6%" in _section()


def test_the_balanced_figure_is_four_times_the_pooled_one():
    data = _measured()["out_of_domain"]
    balanced = data["balanced"]
    assert balanced["languages"] == 8
    assert balanced["accuracy"] == pytest.approx(PUBLISHED_BALANCED_ACC, abs=5e-5)
    assert balanced["ece"] == pytest.approx(PUBLISHED_BALANCED_ECE, abs=5e-5)
    assert balanced["ece"] > data["pooled_uncalibrated"]["ece"] * 4


def test_the_signed_gap_is_far_smaller_than_the_absolute_one():
    """The cancellation, which is the section's central claim.

    A small signed mean beside a large absolute mean is what it looks like when two
    groups fail in opposite directions. Without this the balanced ECE would read as
    uniform mild miscalibration.
    """
    balanced = _measured()["out_of_domain"]["balanced"]
    assert balanced["signed_gap"] == pytest.approx(PUBLISHED_BALANCED_SIGNED, abs=5e-5)
    assert abs(balanced["signed_gap"]) < balanced["ece"] / 5
    # Both directions have to be present, or there is nothing to cancel.
    assert balanced["overconfident_languages"], balanced
    assert balanced["underconfident_languages"], balanced
    assert set(balanced["overconfident_languages"]) == {"pnb", "ur"}


# ------------------------------------------------------ the direction that matters


def test_the_overconfidence_lands_on_the_language_the_model_is_worst_at():
    """If this inverted, section 5.1's conclusion about abstention would be wrong."""
    balanced = _measured()["out_of_domain"]["balanced"]
    assert balanced["weakest_language"] == "ur"
    assert balanced["overconfident_where_accuracy_is_worst"] is True
    rows = balanced["per_language"]
    # Every language the model is right about, it is underconfident about.
    for code, row in rows.items():
        if row["accuracy"] > 0.9:
            assert row["overconfidence"] < 0, (code, row)
        else:
            assert row["overconfidence"] > 0, (code, row)


def test_every_per_language_row_matches_the_published_table():
    rows = _measured()["out_of_domain"]["balanced"]["per_language"]
    assert set(rows) == set(PUBLISHED_PER_LANGUAGE)
    for code, (n, accuracy, ece, over) in PUBLISHED_PER_LANGUAGE.items():
        row = rows[code]
        assert row["n"] == n, code
        assert row["accuracy"] == pytest.approx(accuracy, abs=5e-5), code
        assert row["ece"] == pytest.approx(ece, abs=5e-5), code
        assert row["overconfidence"] == pytest.approx(over, abs=5e-5), code


def test_the_out_of_domain_urdu_accuracy_agrees_with_section_5():
    """Two scripts, one corpus: a disagreement would mean one of them is wrong."""
    mine = _measured()["out_of_domain"]["balanced"]["per_language"]["ur"]
    other = ROOT / "research" / "results" / "out_of_domain.json"
    if not other.is_file():
        pytest.skip("research/results/out_of_domain.json is not present")
    published = json.loads(other.read_text(encoding="utf-8"))["per_language"]["ur"]
    assert mine["n"] == published["n"]
    assert mine["accuracy"] == pytest.approx(published["correct"] / published["n"], abs=1e-4)


def test_the_in_domain_temperature_makes_things_worse_off_domain():
    data = _measured()["out_of_domain"]
    assert data["pooled_calibrated"]["ece"] == pytest.approx(PUBLISHED_POOLED_ECE_SCALED, abs=5e-5)
    assert data["pooled_calibrated"]["ece"] > data["pooled_uncalibrated"]["ece"] * 5
    # Overcorrected: overconfident before, underconfident after.
    assert data["pooled_uncalibrated"]["overconfidence"] > 0
    assert data["pooled_calibrated"]["overconfidence"] < 0


# ----------------------------------------------------------------- the prose itself


def test_every_figure_in_the_section_comes_from_the_artefact():
    data = _measured()
    section = _section()
    for value in (
        f"{PUBLISHED_TEMPERATURE}",
        f"{data['in_domain']['uncalibrated']['ece']:.4f}",
        f"{data['in_domain']['calibrated']['ece']:.4f}",
        f"{data['out_of_domain']['pooled_uncalibrated']['ece']:.4f}",
        f"{data['out_of_domain']['pooled_calibrated']['ece']:.4f}",
        f"{data['out_of_domain']['balanced']['ece']:.4f}",
        f"{data['out_of_domain']['balanced']['accuracy']:.4f}",
        f"{data['out_of_domain']['pooled_uncalibrated']['n']:,}",
    ):
        assert value in section, value


def test_the_section_does_not_recommend_shipping_a_probability():
    """The conclusion the measurement supports; an edit must not soften it upward."""
    section = _section()
    assert "does not license shipping a probability" in section
    assert "anti-correlated" in section
    assert "calibrate per language, not globally" in section
    # And it must keep saying the pooled figure is the misleading one.
    assert "that figure is worthless" in section


def test_the_new_references_are_cited_and_defined():
    text = _repo_file(PAPER)
    for key in ("@guo2017", "@ovadia2019"):
        assert text.count(f"[{key}]") >= 1, key
    assert "Temperature scaling" in text
    assert "Dataset Shift" in text


README = ROOT / "README.md"


def _readme_section() -> str:
    text = _repo_file(README)
    start = text.index("### The confidence is anti-correlated with competence off Wikipedia")
    end = text.index("**Speed.** `import urdunlp`", start)
    return " ".join(text[start:end].split())


def test_the_readme_repeats_every_number_the_paper_does():
    """Two copies of one table is two chances to drift; nothing else compares them."""
    data = _measured()
    section = _readme_section()
    balanced = data["out_of_domain"]["balanced"]
    for code, row in balanced["per_language"].items():
        assert f"| {code} |" in _repo_file(README), code
        assert f"{row['accuracy']:.4f}" in section, (code, row["accuracy"])
        assert f"{row['ece']:.4f}" in section, (code, row["ece"])
    for value in (
        f"{balanced['ece']:.4f}",
        f"{balanced['accuracy']:.4f}",
        f"{balanced['signed_gap']:.4f}",
        f"{data['in_domain']['uncalibrated']['ece']:.4f}",
        f"{data['in_domain']['calibrated']['ece']:.4f}",
        f"{data['out_of_domain']['pooled_uncalibrated']['ece']:.4f}",
        f"{data['out_of_domain']['pooled_calibrated']['ece']:.4f}",
        f"{PUBLISHED_TEMPERATURE}",
    ):
        assert value in section, value


def test_the_readme_explains_why_there_is_no_probability_field():
    """The API decision the measurement supports, kept attached to its reason."""
    section = _readme_section()
    assert "no `probability` field" in section
    assert "anti-correlated with competence" in section
    assert "opposite sign" in section


def test_the_guess_still_exposes_no_probability():
    """The README's claim about the API, checked against the API."""
    from urdunlp import identify_language

    guess = identify_language("یہ کتاب میری ہے")
    assert not hasattr(guess, "probability"), (
        "a probability field was added; README and section 5.1 argue against one and "
        "would both need rewriting"
    )
    assert isinstance(guess.margin, float)
