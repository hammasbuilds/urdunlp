"""Sections 5.2 and 5.3: the remedy section 5.1 named, and the curve section 4.5 drew.

Both sections exist to say that a measurement which looks like a success is not one, so the
tests here are mostly about *pairs* of numbers whose relationship carries the claim rather
than about the numbers themselves:

  * §5.2 holds only if the pooled ECE improves enormously while the balanced one does not.
    Either half alone is a different paper. A later edit that reported the pooled column
    only would read as "per-language calibration fixes it", which is what the section
    denies.
  * §5.3 holds only if the off-domain retention curve looks *better* than the in-domain one
    while the kept set loses the languages the toolkit exists for. Again, either half alone
    inverts the conclusion.

Two further properties are asserted because nothing in the arithmetic enforces them and
both would be silent if they broke: no arm may change a prediction (temperature scaling is
order-preserving, so every arm must report identical accuracy), and Shahmukhi Punjabi's
temperature must be fitted on so few rows that the section's explanation of why it gets
worse is the explanation and not a guess.
"""

from __future__ import annotations

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MEASURED = ROOT / "research" / "results" / "calibration.json"
PAPER = ROOT / "paper" / "main.md"
README = ROOT / "README.md"

ARMS = ("global", "oracle", "predicted")

# As section 5.2 publishes them.
PUBLISHED_POOLED_ECE = {"global": 0.0248, "oracle": 0.0015, "predicted": 0.0031}
PUBLISHED_BALANCED_ECE = {"global": 0.0528, "oracle": 0.0468, "predicted": 0.0547}
PUBLISHED_PER_LANGUAGE_ECE = {
    "ar": (0.0281, 0.0016, 0.0041),
    "fa": (0.0449, 0.0015, 0.0043),
    "ckb": (0.0024, 0.0011, 0.0010),
    "ug": (0.0004, 0.0002, 0.0002),
    "ur": (0.0934, 0.0620, 0.0737),
    "skr": (0.0753, 0.0730, 0.0391),
    "pnb": (0.1254, 0.1885, 0.2607),
}
PUBLISHED_FITTED_TEMPERATURE = {
    "ar": 1.0356,
    "fa": 0.8577,
    "ckb": 1.5859,
    "ug": 1.1809,
    "ur": 2.2426,
    "skr": 0.9675,
    "pnb": 2.1426,
}
PUBLISHED_EVAL_N = {
    "ar": 32_021,
    "fa": 15_719,
    "ckb": 6_246,
    "ug": 4_958,
    "ur": 1_406,
    "skr": 67,
    "pnb": 35,
}
PUBLISHED_MEASURED_ON = 60_473

# As section 5.3 publishes them: coverage -> (in-domain acc, off-domain acc).
PUBLISHED_CURVE = {
    1.0: (0.9229, 0.9785),
    0.9: (0.9677, 0.9991),
    0.75: (0.9984, 0.9998),
    0.5: (0.9994, 0.9999),
}
PUBLISHED_URDU_SHARE = {1.0: 0.0233, 0.9: 0.0012, 0.75: 0.0001, 0.5: 0.0001}
PUBLISHED_HALF_COVERAGE = {
    "ks": (1.0, 1.0, None),
    # language -> (full accuracy, half accuracy, error share remaining at half)
    "ar": (0.9861, 1.0000, 0.0),
    "fa": (0.9895, 0.9997, 0.0121),
    "ckb": (0.9982, 1.0000, 0.0),
    "ug": (0.9998, 1.0000, 0.0),
    "ur": (0.5302, 0.5946, 0.4315),
    "skr": (0.9474, 1.0000, 0.0),
    "pnb": (0.6857, 0.8000, 0.3182),
}


def _measured() -> dict:
    if not MEASURED.exists():
        pytest.skip("research/results/calibration.json is not present")
    return json.loads(MEASURED.read_text(encoding="utf-8"))


def _repo_file(path: pathlib.Path) -> str:
    if not path.is_file():
        pytest.skip(f"{path.name} is not present - running against an installed wheel")
    return path.read_text(encoding="utf-8")


def _arms() -> dict:
    payload = _measured()
    if "per_language_temperature" not in payload:
        pytest.skip("calibration.json predates the per-language arms")
    return payload["per_language_temperature"]


def _curve() -> dict:
    payload = _measured()
    if "selective_prediction" not in payload:
        pytest.skip("calibration.json predates the retention curve")
    return payload["selective_prediction"]


def _by_coverage(rows: list[dict]) -> dict:
    return {row["coverage"]: row for row in rows}


def _section(heading: str, until: str) -> str:
    text = _repo_file(PAPER)
    start = text.index(heading)
    return " ".join(text[start : text.index(until, start)].split())


# --------------------------------------------------------- 5.2, the remedy that is not one


def test_the_arms_were_fitted_and_measured_on_different_halves():
    arms = _arms()
    assert arms["stratified_by_language"] is True
    assert arms["measured_on"] == PUBLISHED_MEASURED_ON
    # Halves, so neither is a small corner of the other. An arm fitted on nine tenths and
    # measured on one would report a tighter fit for a reason that is not the method.
    assert abs(arms["fitted_on"] - arms["measured_on"]) <= len(arms["per_language"]) + 1


def test_the_pooled_figure_says_the_remedy_worked():
    arms = _arms()
    for name, published in PUBLISHED_POOLED_ECE.items():
        assert arms[name]["ece"] == pytest.approx(published, abs=5e-5), name
    # The size of the apparent win, which is what makes the pooled column dangerous.
    assert arms["global"]["ece"] / arms["oracle"]["ece"] > 10


def test_the_balanced_figure_says_it_did_not():
    arms = _arms()
    for name, published in PUBLISHED_BALANCED_ECE.items():
        assert arms[name]["balanced_ece"] == pytest.approx(published, abs=5e-5), name
    # The claim of the section, as a relationship rather than as three numbers: the oracle
    # removes most of the pooled error and little of the balanced one. Compared as
    # *relative* reductions, because the two are errors over different populations and the
    # absolute difference between them is not a quantity about anything. An earlier version
    # of this test asserted a factor of ten on the absolute gains and was simply false -
    # the absolute balanced gain is the larger of the two.
    pooled = 1 - arms["oracle"]["ece"] / arms["global"]["ece"]
    balanced = 1 - arms["oracle"]["balanced_ece"] / arms["global"]["balanced_ece"]
    assert pooled > 0.9, pooled
    assert balanced < 0.2, balanced
    assert pooled > 5 * balanced, (pooled, balanced)


def test_the_deployable_arm_is_worse_on_the_balanced_figure_as_published():
    """As published over all seven languages - a figure the next test shows is fragile.

    Kept because the published number has to be the artefact's, and named so that a reader
    of this file does not take it for a robust finding. The leave-one-out test below is the
    one that says what it is worth.
    """
    arms = _arms()
    assert arms["predicted"]["balanced_ece"] > arms["global"]["balanced_ece"]
    assert arms["predicted"]["ece"] < arms["global"]["ece"], (
        "if the deployable arm were worse on the pooled figure too, the section's point "
        "about which column a reader quotes would not arise"
    )


def test_neither_direction_in_the_balanced_column_survives_dropping_one_language():
    """The finding the section rests on, and the reason the previous test is hedged.

    A balanced mean gives each language one vote, so it is dominated by its worst group -
    and two of these seven hold 35 and 67 evaluation rows. Both arms' advantages over the
    global temperature are decided by one language each, which is a fact about the
    population and not about temperature scaling.
    """
    arms = _arms()
    if "leave_one_out" not in arms:
        pytest.skip("calibration.json predates the leave-one-out sensitivity")
    loo = arms["leave_one_out"]
    assert loo["drops_that_reverse_predicted_against_global"] == ["pnb"]
    assert loo["drops_that_reverse_oracle_against_global"] == ["fa"]
    # Without Punjabi the deployable arm halves the balanced error rather than raising it.
    without_pnb = loo["dropping_one"]["pnb"]
    assert without_pnb["predicted"] < without_pnb["global"] / 1.5, without_pnb
    # And the published row is the one with every language in it.
    assert loo["all_languages"]["predicted"] == pytest.approx(
        arms["predicted"]["balanced_ece"], abs=5e-5
    )


def test_the_language_that_decides_it_is_the_one_with_almost_no_rows():
    """Named, so that "fragile" is attributable rather than a hedge."""
    arms = _arms()
    if "leave_one_out" not in arms:
        pytest.skip("calibration.json predates the leave-one-out sensitivity")
    loo = arms["leave_one_out"]
    sizes = arms["eval_per_language_n"]
    for code in loo["drops_that_reverse_predicted_against_global"]:
        assert sizes[code] < 100, (code, sizes[code])
    # Every language is dropped in turn, not a chosen few.
    assert set(loo["dropping_one"]) == set(arms["global"]["per_language_ece"])


def test_the_sensitivity_table_in_the_section_is_the_artefact_s():
    arms = _arms()
    if "leave_one_out" not in arms:
        pytest.skip("calibration.json predates the leave-one-out sensitivity")
    section = _section("## 5.2 Per-language calibration is not the remedy", "\n## 5.3 ")
    for dropped, means in arms["leave_one_out"]["dropping_one"].items():
        for arm, value in means.items():
            assert f"{value:.4f}" in section, (dropped, arm, value)


def test_the_section_says_the_balanced_comparison_is_not_robust():
    section = _section("## 5.2 Per-language calibration is not the remedy", "\n## 5.3 ")
    assert "survives dropping one language" in section
    assert "too fragile" in section
    # And it names what is robust instead, rather than leaving a hole.
    assert "is* robust" in section or "is robust" in section


def test_no_arm_changes_a_single_prediction():
    """Temperature scaling is order-preserving, per item, whatever the temperature.

    So all three arms must report identical accuracy. An arm that did not would mean the
    per-item scaling had been applied to the ranking rather than to the posterior, and the
    section's closing sentence - that calibration was never going to make the model right -
    would be false.
    """
    arms = _arms()
    accuracies = {arms[name]["accuracy"] for name in ARMS}
    assert len(accuracies) == 1, {name: arms[name]["accuracy"] for name in ARMS}


def test_punjabi_gets_worse_under_both_arms_and_the_reason_is_in_the_row():
    arms = _arms()
    ece = arms["global"]["per_language_ece"]["pnb"]
    assert arms["oracle"]["per_language_ece"]["pnb"] > ece
    assert arms["predicted"]["per_language_ece"]["pnb"] > arms["oracle"]["per_language_ece"]["pnb"]
    # The explanation the section gives, asserted rather than assumed: the temperature that
    # made it worse was fitted on fewer than forty sentences.
    assert arms["per_language"]["pnb"]["n"] < 40, arms["per_language"]["pnb"]


def test_every_per_language_cell_of_the_table_matches():
    arms = _arms()
    for code, published in PUBLISHED_PER_LANGUAGE_ECE.items():
        got = tuple(arms[name]["per_language_ece"][code] for name in ARMS)
        assert got == pytest.approx(published, abs=5e-5), code
    for code, temperature in PUBLISHED_FITTED_TEMPERATURE.items():
        assert arms["per_language"][code]["temperature"] == pytest.approx(temperature, abs=5e-5)
        assert not arms["per_language"][code]["at_the_bracket_edge"], code
    for code, n in PUBLISHED_EVAL_N.items():
        assert arms["eval_per_language_n"][code] == n, code


def test_the_items_without_a_temperature_are_counted_rather_than_dropped():
    """Dropping them would restrict the measurement to the languages the arm covered."""
    arms = _arms()
    assert arms["global"]["items_on_the_global_temperature"] == 0
    assert arms["predicted"]["items_on_the_global_temperature"] > 0
    # Every arm scores the same population, which is what makes the columns comparable.
    assert len({arms[name]["n"] for name in ARMS}) == 1


# ------------------------------------------------- 5.3, abstention that drops the language


def test_the_off_domain_curve_looks_better_than_the_in_domain_one():
    curve = _curve()
    inside = _by_coverage(curve["in_domain"])
    outside = _by_coverage(curve["out_of_domain"])
    for coverage, (published_in, published_out) in PUBLISHED_CURVE.items():
        assert inside[coverage]["accuracy"] == pytest.approx(published_in, abs=5e-5), coverage
        assert outside[coverage]["accuracy"] == pytest.approx(published_out, abs=5e-5), coverage
    # The trap, as a relationship: off domain is better at every published coverage.
    for coverage in PUBLISHED_CURVE:
        assert outside[coverage]["accuracy"] >= inside[coverage]["accuracy"], coverage
    # And dropping the least confident tenth appears to remove almost every error.
    assert outside[0.9]["error_share_remaining"] < 0.05


def test_what_the_threshold_actually_dropped_was_the_languages():
    curve = _curve()
    outside = _by_coverage(curve["out_of_domain"])
    for coverage, share in PUBLISHED_URDU_SHARE.items():
        got = outside[coverage]["share_of_kept_by_language"].get("ur", 0.0)
        assert got == pytest.approx(share, abs=5e-5), coverage
    # 95% of the Urdu items are gone in the first tenth dropped.
    full = outside[1.0]["share_of_kept_by_language"]["ur"] * outside[1.0]["kept"]
    kept = outside[0.9]["share_of_kept_by_language"]["ur"] * outside[0.9]["kept"]
    assert kept / full < 0.1, (full, kept)
    # Punjabi is gone entirely by three-quarter coverage.
    assert outside[0.75]["share_of_kept_by_language"].get("pnb", 0.0) == 0.0


def test_inside_each_language_abstention_keeps_the_errors_it_should_drop():
    curve = _curve()
    rows = curve["out_of_domain_per_language"]
    for code, (full, half, remaining) in PUBLISHED_HALF_COVERAGE.items():
        row = rows[code]
        assert row["accuracy_at_full_coverage"] == pytest.approx(full, abs=5e-5), code
        assert row["accuracy_at_half_coverage"] == pytest.approx(half, abs=5e-5), code
        if remaining is None:
            assert row["error_share_remaining_at_half"] is None, code
        else:
            assert row["error_share_remaining_at_half"] == pytest.approx(remaining, abs=5e-5), code
    # The sentence the section rests on: half the Urdu thrown away, and most of its errors
    # still there. In domain the same operation left 0.4%.
    assert rows["ur"]["error_share_remaining_at_half"] > 0.4
    assert rows["pnb"]["error_share_remaining_at_half"] > 0.3
    assert _by_coverage(curve["in_domain"])[0.5]["error_share_remaining"] < 0.01


def test_a_language_with_too_few_rows_gets_no_rate_of_its_own():
    """Sindhi has four off-domain sentences, so it gets no per-language curve.

    Kashmiri has 34, which clears the 30-row minimum, so it does get one - and its error
    share at half coverage is `None` rather than 0.0, because it made no errors to leave
    behind. A zero there would read as "abstention removed every error", which is the
    section's conclusion for the easy languages and not something this row establishes.
    """
    rows = _curve()["out_of_domain_per_language"]
    assert "sd" not in rows
    assert rows["ks"]["n"] == 34
    assert rows["ks"]["error_share_remaining_at_half"] is None


# ------------------------------------------------------------------------------- the prose


def test_every_figure_in_section_5_2_comes_from_the_artefact():
    arms = _arms()
    section = _section("## 5.2 Per-language calibration is not the remedy", "\n## 5.3 ")
    for name in ARMS:
        assert f"{arms[name]['ece']:.4f}" in section, name
        assert f"{arms[name]['balanced_ece']:.4f}" in section, name
    for code in PUBLISHED_PER_LANGUAGE_ECE:
        assert f"{arms['per_language'][code]['temperature']:.4f}" in section, code
    assert f"{arms['measured_on']:,}" in section


def test_every_figure_in_section_5_3_comes_from_the_artefact():
    curve = _curve()
    section = _section("## 5.3 Abstention off domain discards the languages", "\n# 6. ")
    outside = _by_coverage(curve["out_of_domain"])
    inside = _by_coverage(curve["in_domain"])
    for coverage in PUBLISHED_CURVE:
        assert f"{outside[coverage]['accuracy']:.4f}" in section, coverage
        assert f"{inside[coverage]['accuracy']:.4f}" in section, coverage
    rows = curve["out_of_domain_per_language"]
    # Quoted at the precision the artefact stores, which is four places. The prose said
    # 43.2% - the half-up rounding of 43.15% - and the artefact does not hold the number to
    # the precision that rounding needs.
    assert f"{rows['ur']['error_share_remaining_at_half'] * 100:.2f}%" in section
    assert f"{rows['pnb']['error_share_remaining_at_half'] * 100:.2f}%" in section


def test_the_oracle_arm_is_labelled_undeployable_wherever_it_is_quoted():
    """Its number is the best in the table and it cannot be shipped."""
    section = _section("## 5.2 Per-language calibration is not the remedy", "\n## 5.3 ")
    assert "not deployable" in section or "Not deployable" in section
    assert "knowing the language is the task" in section


def test_section_5_1_no_longer_offers_the_remedy_without_the_result():
    """It used to end by naming per-language calibration as what a future version would do."""
    section = _section("## 5.1 The confidence does not notice", "\n## 5.2 ")
    assert "calibrate per language" in section
    assert "5.2" in section, (
        "5.1 points at the remedy; without a pointer to the section that measured it, the "
        "paper recommends something it elsewhere refutes"
    )


def test_section_4_5_points_at_the_curve_drawn_off_domain():
    section = _section("## 4.5 Knowing when to doubt the answer", "\n## 4.6 ")
    assert "5.3" in section
    rows = _curve()["out_of_domain_per_language"]
    assert f"{rows['ur']['error_share_remaining_at_half'] * 100:.2f}%" in section


def test_the_readme_carries_the_negative_result_too():
    text = _repo_file(README)
    arms = _arms()
    assert "per-language" in text.lower()
    assert f"{arms['predicted']['balanced_ece']:.4f}" in text
    assert f"{arms['oracle']['ece']:.4f}" in text
