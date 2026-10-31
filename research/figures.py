"""Every figure in the paper, one function each.

House rules, so the set reads as one system:

  * matplotlib only, no seaborn - one fewer dependency and no hidden style state.
  * a colourblind-safe palette, and nothing encoded by colour alone: every series is also
    distinguishable by marker, hatch or position.
  * PDF for the manuscript and PNG for the README, written side by side from one call.
  * no chart junk: no gridlines behind bars, no 3D, no dual axes, no truncated y-axis on a
    bar chart. A truncated axis on a *line* chart is labelled as truncated.
  * every figure states its n in the caption text returned by its function, so a figure
    cannot reach the paper without its denominator.

Each function returns (path_pdf, path_png, caption) and writes nothing else.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on this machine; set before pyplot is imported
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "research" / "figures"

# Okabe-Ito, which is colourblind-safe and prints legibly in grey.
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#E69F00", "#009E73", "#999999"
TOOL_COLOUR = {"urdunlp": BLUE, "fasttext": ORANGE, "langdetect": GREEN}
TOOL_MARKER = {"urdunlp": "o", "fasttext": "s", "langdetect": "^"}
TOOL_HATCH = {"urdunlp": "", "fasttext": "//", "langdetect": ".."}
TOOL_NAME = {"urdunlp": "urdunlp", "fasttext": "fastText lid.176",
             "langdetect": "langdetect"}

# The eleven, in the order the paper uses: Urdu first, then the languages it is confused
# with, then the rest of the script family. Not alphabetical - the ordering is the
# argument, because the confusion is concentrated in the first six.
ORDER = ["ur", "pnb", "skr", "sd", "ps", "ks", "fa", "ar", "ckb", "ug", "azb"]
ENGLISH = {
    "ur": "Urdu", "pnb": "Punjabi (Shahmukhi)", "skr": "Saraiki", "sd": "Sindhi",
    "ps": "Pashto", "ks": "Kashmiri", "fa": "Persian", "ar": "Arabic",
    "ckb": "Sorani Kurdish", "ug": "Uyghur", "azb": "South Azerbaijani",
}


def _save(fig, name: str) -> tuple[Path, Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf, png = FIGURES / f"{name}.pdf", FIGURES / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return pdf, png


def _bare(ax) -> None:
    """Drop the furniture that carries no information."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# --- 1. coverage ---------------------------------------------------------------------


def coverage_figure(coverages: dict[str, dict]) -> tuple[Path, Path, str]:
    """Which of the eleven classes each system can name at all.

    The paper's first figure, because it decides how to read every accuracy after it: a
    detector with no Saraiki label is not wrong about Saraiki, it was asked a question it
    cannot answer.
    """
    tools = ["urdunlp", "fasttext", "langdetect"]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for row, tool in enumerate(tools):
        known = set(coverages[tool]["has_a_label_for"])
        for col, code in enumerate(ORDER):
            has = code in known
            ax.add_patch(plt.Rectangle(
                (col, row), 1, 1,
                facecolor=TOOL_COLOUR[tool] if has else "white",
                edgecolor=GREY, linewidth=0.6,
                hatch="" if has else "xx",
            ))
            if not has:
                ax.text(col + 0.5, row + 0.5, "—", ha="center", va="center",
                        color=GREY, fontsize=9)
    ax.set_xlim(0, len(ORDER))
    ax.set_ylim(0, len(tools))
    ax.set_xticks([i + 0.5 for i in range(len(ORDER))])
    ax.set_xticklabels(ORDER, fontsize=9)
    ax.set_yticks([i + 0.5 for i in range(len(tools))])
    ax.set_yticklabels([TOOL_NAME[t] for t in tools], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("language in the task", fontsize=9)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    counts = {t: coverages[t]["covered"] for t in tools}
    caption = (
        "Label coverage of the eleven-language task. A filled cell means the system has a "
        f"label for that language: urdunlp {counts['urdunlp']}/11, "
        f"fastText {counts['fasttext']}/11, langdetect {counts['langdetect']}/11. "
        "Hatched cells are languages the system cannot name, so no accuracy over them is "
        "meaningful."
    )
    return (*_save(fig, "fig1_coverage"), caption)


# --- 2. accuracy against text length -------------------------------------------------


def length_figure(by_length: dict, restricted: dict | None = None) -> tuple[Path, Path, str]:
    """Accuracy as the text gets shorter, with the restricted reading shown too.

    Two lines per baseline: over all eleven classes (solid) and over only the classes it
    can name (dashed). Plotting one without the other is how this comparison would
    mislead in either direction.
    """
    labels = list(by_length)
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for tool in ("urdunlp", "fasttext", "langdetect"):
        ys = [by_length[length][tool]["accuracy_all_languages"] for length in labels]
        ax.plot(x, ys, marker=TOOL_MARKER[tool], color=TOOL_COLOUR[tool],
                linewidth=2, label=f"{TOOL_NAME[tool]}, all 11")
        if tool != "urdunlp":
            yr = [by_length[length][tool]["accuracy_where_it_has_a_label"]
                  for length in labels]
            ax.plot(x, yr, marker=TOOL_MARKER[tool], color=TOOL_COLOUR[tool],
                    linewidth=1.3, linestyle="--", alpha=0.85,
                    label=f"{TOOL_NAME[tool]}, where labelled")
    ax.set_xticks(list(x))
    ax.set_xticklabels([l.replace(" chars", " characters") for l in labels], fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("accuracy", fontsize=10)
    ax.set_xlabel("text available to the classifier", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    _bare(ax)
    n = by_length[labels[0]]["n"]
    caption = (
        f"Accuracy against the amount of text, {n:,} test paragraphs per point. Solid "
        "lines are accuracy over all eleven languages; dashed lines restrict each baseline "
        "to the languages it has a label for. The y-axis starts at zero and is not "
        "truncated."
    )
    return (*_save(fig, "fig2_accuracy_by_length"), caption)


# --- 3. per-class F1 -----------------------------------------------------------------


def per_class_figure(scores: dict[str, dict]) -> tuple[Path, Path, str]:
    """F1 per language per system, grouped by language.

    A bar is absent, not zero, where the system has no label for that class - and the gap
    is annotated, because an absent bar and a zero bar mean different things and a reader
    cannot tell them apart by eye.
    """
    tools = ["urdunlp", "fasttext", "langdetect"]
    width = 0.26
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    for i, tool in enumerate(tools):
        per = scores[tool]["per_class"]
        xs, ys = [], []
        for j, code in enumerate(ORDER):
            f1 = per.get(code, {}).get("f1")
            if f1 is None:
                ax.text(j + (i - 1) * width, 0.02, "no\nlabel", ha="center", va="bottom",
                        fontsize=6, color=GREY, linespacing=0.9)
                continue
            xs.append(j + (i - 1) * width)
            ys.append(f1)
        ax.bar(xs, ys, width=width, color=TOOL_COLOUR[tool], label=TOOL_NAME[tool],
               hatch=TOOL_HATCH[tool], edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1", fontsize=10)
    ax.set_xlabel("language", fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center")
    _bare(ax)
    macro = {t: scores[t]["macro_f1"] for t in tools}
    caption = (
        "Per-language F1 on whole paragraphs. Macro-F1 over the classes each system can "
        f"score: urdunlp {macro['urdunlp']:.3f} over {scores['urdunlp']['macro_f1_over']}, "
        f"fastText {macro['fasttext']:.3f} over {scores['fasttext']['macro_f1_over']}, "
        f"langdetect {macro['langdetect']:.3f} over "
        f"{scores['langdetect']['macro_f1_over']}. A missing bar is a class the system has "
        "no label for, which is not the same as an F1 of zero."
    )
    return (*_save(fig, "fig3_per_class_f1"), caption)


# --- 4. where the mass goes ----------------------------------------------------------


def misattribution_figure(rows: dict[str, dict[str, int]],
                          tool: str = "langdetect") -> tuple[Path, Path, str]:
    """For each language a system cannot name, what it answers instead.

    The paper's central figure. A stacked bar per true language, segments ordered by size,
    so "Punjabi is called Urdu" is a length rather than a sentence.
    """
    shown = [c for c in ORDER if c in rows and rows[c]]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    palette = {"ur": BLUE, "fa": ORANGE, "ar": GREEN, "pnb": "#CC79A7"}
    seen: list[str] = []
    for i, code in enumerate(shown):
        total = sum(rows[code].values())
        left = 0.0
        for answer, count in sorted(rows[code].items(), key=lambda kv: -kv[1]):
            share = count / total
            colour = palette.get(answer, GREY)
            ax.barh(i, share, left=left, color=colour, edgecolor="white", linewidth=0.6)
            if share > 0.08:
                ax.text(left + share / 2, i, answer, ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
            left += share
            if answer not in seen:
                seen.append(answer)
    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels([f"{ENGLISH[c]}  (n={sum(rows[c].values())})" for c in shown],
                       fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of that language's paragraphs", fontsize=10)
    ax.legend(handles=[Patch(facecolor=palette.get(a, GREY), label=a) for a in seen],
              fontsize=8, frameon=False, ncol=len(seen), loc="upper center",
              bbox_to_anchor=(0.5, 1.12))
    _bare(ax)
    worst = max(shown, key=lambda c: rows[c].get("ur", 0) / sum(rows[c].values()))
    share = rows[worst].get("ur", 0) / sum(rows[worst].values())
    caption = (
        f"What {TOOL_NAME[tool]} answers for the languages it has no label for, whole "
        f"paragraphs. {ENGLISH[worst]} is labelled Urdu in {share:.1%} of its paragraphs. "
        "Each bar sums to one; segments are the answers the system actually gave."
    )
    return (*_save(fig, f"fig4_misattribution_{tool}"), caption)


# --- 5. confusion matrix -------------------------------------------------------------


def confusion_figure(rows: dict[str, dict[str, int]], tool: str,
                     number: int = 5) -> tuple[Path, Path, str]:
    """Row-normalised confusion over the eleven, with off-task answers in one column.

    Row-normalised, because the question is "what happens to Saraiki", not "how much
    Saraiki is there" - the classes are near-balanced, so the raw counts would say the
    same thing less legibly.
    """
    columns = list(ORDER)
    off = sorted({a for r in rows.values() for a in r} - set(ORDER))
    if off:
        columns = columns + ["other"]
    grid = []
    for code in ORDER:
        total = sum(rows.get(code, {}).values()) or 1
        line = []
        for col in ORDER:
            line.append(rows.get(code, {}).get(col, 0) / total)
        if off:
            line.append(sum(rows.get(code, {}).get(a, 0) for a in off) / total)
        grid.append(line)

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    image = ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels(ORDER, fontsize=8)
    ax.set_xlabel("predicted", fontsize=10)
    ax.set_ylabel("true", fontsize=10)
    for i, line in enumerate(grid):
        for j, value in enumerate(line):
            if value >= 0.01:
                ax.text(j, i, f"{value:.2f}".lstrip("0") if value < 1 else "1",
                        ha="center", va="center", fontsize=6.5,
                        color="white" if value > 0.55 else "#222222")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03,
                 label="share of the true language's paragraphs")
    total = sum(sum(r.values()) for r in rows.values())
    caption = (
        f"Row-normalised confusion for {TOOL_NAME[tool]} on whole paragraphs, "
        f"{total:,} paragraphs. Each row sums to one. "
        + ("The `other` column collects answers outside the task's eleven labels: "
           + ", ".join(off) + "." if off else "Every answer fell inside the task's labels.")
    )
    return (*_save(fig, f"fig{number}_confusion_{tool}"), caption)


# --- 6. the margin, as a distribution ------------------------------------------------


def margin_figure(deciles: list[dict]) -> tuple[Path, Path, str]:
    """Accuracy per margin decile, with the error count underneath.

    Deciles, not the three fixed bins also in the report: those put 6,131 of 6,136 items
    in one bin, so a chart of them is one point doing all the work and looks flatteringly
    flat. Equal-sized bins put the question where it belongs - does a low margin actually
    mean a likely error.

    The lower panel is the error count, not the item count: the bins are equal by
    construction, so items carry no information and errors carry all of it.
    """
    xs = [d["decile"] for d in deciles]
    accuracies = [d["accuracy"] for d in deciles]
    errors = [d["errors"] for d in deciles]

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(6.6, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.0], "hspace": 0.1},
    )
    top.plot(xs, accuracies, marker="o", color=BLUE, linewidth=2)
    lowest = min(accuracies)
    top.set_ylim(min(0.9, lowest - 0.01), 1.004)
    top.set_ylabel("accuracy", fontsize=10)
    top.annotate(f"{accuracies[0]:.4f}", (xs[0], accuracies[0]),
                 textcoords="offset points", xytext=(6, -2), fontsize=8, color=BLUE)
    _bare(top)
    bottom.bar(xs, errors, color=ORANGE, width=0.65, edgecolor="white", linewidth=0.4)
    for x, e in zip(xs, errors):
        if e:
            bottom.text(x, e, str(e), ha="center", va="bottom", fontsize=7.5)
    bottom.set_ylabel("errors", fontsize=9)
    bottom.set_xticks(xs)
    bottom.set_xlabel("margin decile (1 = narrowest margin)", fontsize=10)
    _bare(bottom)
    total = sum(d["n"] for d in deciles)
    all_errors = sum(errors)
    share = errors[0] / all_errors if all_errors else 0
    caption = (
        f"Accuracy by margin decile on whole test paragraphs, {total:,} items, "
        f"{deciles[0]['n']:,}-{deciles[-1]['n']:,} per decile. "
        f"{errors[0]} of the {all_errors} errors ({share:.0%}) fall in the narrowest "
        "decile, so the margin carries usable information about which answers to doubt. "
        "The accuracy axis is truncated below 0.9 to make the spread visible; the error "
        "panel is not truncated."
    )
    return (*_save(fig, "fig6_margin_deciles"), caption)


def selective_figure(retention: list[dict]) -> tuple[Path, Path, str]:
    """Risk against coverage: abstain on the narrowest margins and see what is left.

    The figure a reviewer of a classifier expects, and the one the three fixed margin bins
    cannot produce. Coverage on the x-axis, accuracy on the kept items on the y, and the
    share of the original errors still present as the second series - because accuracy on
    a shrinking set rises trivially and the error share is what says whether abstention is
    picking the right items to drop.
    """
    keeps = [r["keep"] for r in retention]
    accuracies = [r["accuracy"] for r in retention]
    kept_share = [r["errors_kept_share"] for r in retention]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(keeps, accuracies, marker="o", color=BLUE, linewidth=2,
            label="accuracy on the items kept")
    ax.plot(keeps, kept_share, marker="s", color=ORANGE, linewidth=2, linestyle="--",
            label="share of all errors still present")
    ax.set_xlabel("coverage (share of items answered)", fontsize=10)
    ax.set_ylabel("accuracy / error share", fontsize=10)
    ax.set_ylim(0, 1.03)
    ax.set_xlim(min(keeps) - 0.03, 1.02)
    ax.legend(fontsize=8, frameon=False, loc="center left")
    _bare(ax)
    total_errors = retention[0]["errors_total"]
    at_half = next((r for r in retention if r["keep"] == 0.5), retention[0])
    caption = (
        "Selective prediction: answer only the items whose margin is in the top fraction. "
        f"At {at_half['keep']:.0%} coverage, accuracy on the {at_half['n']:,} items kept "
        f"is {at_half['accuracy']:.4f} and {at_half['errors_kept']} of the "
        f"{total_errors} errors remain. The error share is plotted beside accuracy "
        "because accuracy on a shrinking set rises whether or not abstention is choosing "
        "the right items."
    )
    return (*_save(fig, "fig7_selective_prediction"), caption)
