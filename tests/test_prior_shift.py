"""Prior-shifted evaluation: the arithmetic, and the numbers section 5 publishes.

The comparison in section 5 is over a roughly balanced test set. Text does not arrive
balanced, and the direction of the distortion is not neutral: a balanced set is the one
on which a model that handles only the three largest languages looks worst, and the one
on which a model tuned for the rare languages looks best. `research/prior_shift.py`
recomputes the comparison under two other priors without re-predicting anything.

The arithmetic is checked against cases whose answer can be worked out by hand before the
published numbers are pinned, because a reweighting that is subtly wrong produces numbers
that look plausible and move in the right direction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "research" / "results"
SHIFTED = RESULTS / "prior_shift.json"
CONFUSION = RESULTS / "confusion_whole.json"

sys.path.insert(0, str(ROOT / "research"))

# Imported here, under a guard, rather than inside each test. CI's second job copies
# `tests/` alone next to an installed distribution, where `research/` does not exist, and
# a bare `import prior_shift` inside a test body is a hard failure in that job rather
# than a skip. This file had that fault on the day it was written, which is why the
# guard is at the top where it can be seen.
prior_shift = pytest.importorskip("prior_shift", reason="research/ is not shipped with the tests")
reweight = prior_shift.reweight


@pytest.fixture(scope="module")
def shifted() -> dict:
    if not SHIFTED.exists():
        pytest.skip("research/results/prior_shift.json has not been generated")
    return json.loads(SHIFTED.read_text(encoding="utf-8"))


# --- the arithmetic -------------------------------------------------------------------


def test_a_uniform_prior_reproduces_the_plain_accuracy():
    """The reweighting must be the identity when the prior is the one already there."""

    rows = {"a": {"a": 8, "b": 2}, "b": {"a": 1, "b": 9}}
    out = reweight(rows, {"a": 0.5, "b": 0.5})
    assert out["accuracy"] == pytest.approx((0.8 + 0.9) / 2)


def test_a_skewed_prior_moves_accuracy_towards_the_majority_class():

    rows = {"a": {"a": 10, "b": 0}, "b": {"a": 10, "b": 0}}  # perfect on a, hopeless on b
    assert reweight(rows, {"a": 0.5, "b": 0.5})["accuracy"] == pytest.approx(0.5)
    assert reweight(rows, {"a": 0.99, "b": 0.01})["accuracy"] == pytest.approx(0.99)
    assert reweight(rows, {"a": 0.01, "b": 0.99})["accuracy"] == pytest.approx(0.01)


def test_precision_on_a_rare_class_collapses_when_a_large_class_leaks_into_it():
    """The reason a balanced benchmark flatters a rare-language classifier.

    Worked by hand: class `big` is 99% of the population and 1% of it is misread as
    `rare`; `rare` is 1% of the population and is read correctly every time. The
    predictions labelled `rare` are then 0.01 from `rare` and 0.99*0.01 = 0.0099 from
    `big`, so precision is 0.01 / 0.0199 = 0.503 - against 0.99 on a balanced set. No
    model changed; the population did.
    """

    rows = {"big": {"big": 99, "rare": 1}, "rare": {"rare": 100}}
    balanced = reweight(rows, {"big": 0.5, "rare": 0.5})["per_class"]["rare"]["precision"]
    skewed = reweight(rows, {"big": 0.99, "rare": 0.01})["per_class"]["rare"]["precision"]
    assert balanced == pytest.approx(0.9901, abs=5e-4)
    assert skewed == pytest.approx(0.5025, abs=5e-4)
    assert skewed < balanced / 1.9


def test_recall_is_unchanged_by_the_prior():
    """The prior-shift assumption, as an assertion about what the code does.

    Holding class-conditional recall fixed is the whole method. If a change of prior
    moved recall, the reweighting would be computing something else.
    """

    rows = {"a": {"a": 7, "b": 3}, "b": {"a": 2, "b": 8}}
    first = reweight(rows, {"a": 0.5, "b": 0.5})["per_class"]
    second = reweight(rows, {"a": 0.95, "b": 0.05})["per_class"]
    for label in ("a", "b"):
        assert first[label]["recall"] == second[label]["recall"]


def test_every_prior_sums_to_one(shifted):
    for name, weights in shifted["prior_weights"].items():
        assert sum(weights.values()) == pytest.approx(1.0), name
        assert set(weights) == set(shifted["labels"]), name


def test_the_crawl_prior_says_which_languages_it_had_to_approximate(shifted):
    """Five of the eleven are absent from CC-100 and that must be on the record."""
    crawl = shifted["priors"]["crawl"]
    assert crawl["approximated"], "a prior with no gaps would need no note"
    for code in crawl["approximated"]:
        assert code in crawl["note"], code
        assert code in shifted["prior_weights"]["crawl"]


# --- the published comparison ---------------------------------------------------------


def test_the_uniform_accuracies_match_the_balanced_confusion_matrices(shifted):
    """The uniform column must agree with the comparison it is derived from."""
    rows_by_system = json.loads(CONFUSION.read_text(encoding="utf-8"))
    for system, body in shifted["systems"].items():
        rows = rows_by_system[system]["rows"]
        recalls = [rows[t].get(t, 0) / sum(rows[t].values()) for t in sorted(rows)]
        assert body["uniform"]["accuracy"] == pytest.approx(
            sum(recalls) / len(recalls), abs=1e-4
        ), system


def test_the_ranking_of_the_competitors_is_not_the_same_under_every_prior(shifted):
    """The finding. If this ever stops holding, section 4.6 has to be rewritten."""
    others = [s for s in shifted["systems"] if s != "urdunlp"]
    orders = {
        prior: tuple(sorted(others, key=lambda s: -shifted["systems"][s][prior]["accuracy"]))
        for prior in shifted["prior_weights"]
    }
    assert len(set(orders.values())) > 1, (
        f"every prior ranks the competitors the same way ({orders}); section 4.6 claims "
        f"the ranking is an artefact of the balanced test set"
    )


def test_langdetect_is_the_case_the_section_is_built_on(shifted):
    """Three languages supported out of eleven: worst on a balanced set, near-perfect
    on a population where those three are 98% of the text."""
    body = shifted["systems"]["langdetect"]
    assert body["uniform"]["accuracy"] < 0.4, body["uniform"]["accuracy"]
    assert body["crawl"]["accuracy"] > 0.9, body["crawl"]["accuracy"]
    supported = {"ar", "fa", "ur"}
    per_class = body["uniform"]["per_class"]
    for label, metrics in per_class.items():
        if label in supported:
            assert metrics["recall"] > 0.9, (label, metrics)
        else:
            assert metrics["recall"] == 0.0, (label, metrics)


def test_this_toolkit_is_the_only_system_strong_under_every_prior(shifted):
    """The claim the section is allowed to make, and no more."""
    for prior in shifted["prior_weights"]:
        ours = shifted["systems"]["urdunlp"][prior]["accuracy"]
        assert ours > 0.98, (prior, ours)
        for system, body in shifted["systems"].items():
            if system != "urdunlp":
                assert body[prior]["accuracy"] <= ours, (prior, system)


def test_the_paper_quotes_these_numbers(shifted):
    paper = (ROOT / "paper" / "main.md").read_text(encoding="utf-8")
    if "## 4.6" not in paper:
        pytest.skip("the prior-shift section has not been written yet")
    section = paper[paper.index("## 4.6") :].split("\n# ")[0]
    for system in ("urdunlp", "pali", "fasttext", "langdetect"):
        for prior in ("uniform", "crawl"):
            value = shifted["systems"][system][prior]["accuracy"]
            assert f"{value:.4f}" in section, (system, prior, value)


def test_the_abstract_quotes_the_two_reweighted_numbers(shifted):
    """The abstract names langdetect's and PALI's crawl-prior accuracies.

    Both are reweightings of numbers that already exist, so nothing re-runs if the
    prior file changes - which is exactly why an abstract can keep quoting a stale one.
    """
    paper = (ROOT / "paper" / "main.md").read_text(encoding="utf-8")
    abstract = paper[paper.index("# Abstract") : paper.index("# 1. ")]
    for system in ("langdetect", "pali"):
        value = shifted["systems"][system]["crawl"]["accuracy"]
        assert f"{value:.4f}" in abstract, (system, value)


def test_the_claims_about_pali_s_per_class_recalls_are_the_measured_ones(shifted):
    """Section 4.6 quotes a range over the eight rare languages and two exceptions.

    An earlier draft said PALI's recalls are "high on the eight rare languages", which
    is true of seven of them and not of Shahmukhi Punjabi at 0.865. A range quoted in
    prose is the kind of claim that stays after the numbers behind it move.
    """
    per_class = shifted["systems"]["pali"]["uniform"]["per_class"]
    rare = {k: v["recall"] for k, v in per_class.items() if k not in {"ar", "fa", "ur"}}
    assert len(rare) == 8, sorted(rare)
    worst = min(rare, key=rare.get)
    others = [v for k, v in rare.items() if k != worst]

    paper = (ROOT / "paper" / "main.md").read_text(encoding="utf-8")
    section = paper[paper.index("## 4.6") :].split("\n# ")[0]
    assert f"{min(others):.3f} to {max(others):.3f}" in section, (min(others), max(others))
    assert f"{rare[worst]:.3f}" in section, rare[worst]
    for label in ("fa", "ur"):
        assert f"{per_class[label]['recall']:.3f}" in section, (label, per_class[label])


def test_the_three_languages_langdetect_supports_really_are_most_of_the_crawl(shifted):
    """Section 4.6 says they are 98% of the text under the CC-100 prior."""
    weights = shifted["prior_weights"]["crawl"]
    share = sum(weights[k] for k in ("ar", "fa", "ur"))
    assert 0.975 < share < 0.99, share
    paper = (ROOT / "paper" / "main.md").read_text(encoding="utf-8")
    section = paper[paper.index("## 4.6") :].split("\n# ")[0]
    assert f"{share * 100:.0f}% of the text" in section, f"{share * 100:.0f}"
