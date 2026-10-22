"""Public coverage-study aggregates/figures and private seed-paired playback."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path

from speaker_lab.duplex.coverage import has_filler
from speaker_lab.duplex.protocol import file_hash, read_json, write_json
from speaker_lab.glossary_metrics import score_pair


def rows(path):
    return [
        json.loads(s)
        for s in path.read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    runs = root / "runs/duplexlab/coverage-v1"
    data = root / "data/duplexlab/coverage-v1"
    out = root / "experiments/duplexlab/reports/2026-10-20-coverage"
    out.mkdir(parents=True, exist_ok=True)
    verification = read_json(runs / "LOCAL_VERIFICATION.json")
    if verification["status"] != "verified":
        raise ValueError("Independent verification required before final reporting")
    selection = read_json(runs / "SELECTION.json")
    comparison = {
        role: read_json(runs / f"COMPARISON-{role}.json") for role in ["dev", "test"]
    }
    training = {}
    for seed in [42, 43, 44]:
        for arm in ["random", "filtered"]:
            record = read_json(runs / f"train-{arm}-{seed}-record/training.json")
            if (
                record["status"] != "trained_and_reloaded"
                or not record["test_complete"]
            ):
                raise ValueError(
                    "Six complete fresh SFTs and actual test reloads required"
                )
            training[f"{arm}-{seed}"] = {
                k: record[k]
                for k in [
                    "status",
                    "optimizer_steps",
                    "epoch",
                    "checkpoint_reloaded",
                    "test_complete",
                    "subprocess_seconds",
                    "checkpoint_hashes",
                    "actual_training_arguments",
                    "export_hash",
                ]
            }
    design = {
        "audit": selection["audit"],
        "duration_seconds": selection["duration_seconds"],
        "quotas": selection["quotas"],
        "pool_marginals": selection["pool_marginals"],
        "source_predictions_sha256": selection["source_predictions_sha256"],
        "scipy": selection["scipy"],
        "reference_token_counts": {
            a: sum(score_pair(r["reference"], "", terms=[])["N"] for r in v)
            for a, v in selection["arms"].items()
        },
        "UNK_counts": {
            a: sum("[UNK]" in r["reference"] for r in v)
            for a, v in selection["arms"].items()
        },
    }
    lo, hi = comparison["test"]["hierarchical_delta_interval"]
    conclusion = (
        "No reliable coverage-selection advantage established; paired interval includes zero."
        if lo <= 0 <= hi
        else "Coverage has lower error in this fixed pilot; generalization beyond this split remains unverified."
        if hi < 0
        else "Coverage has higher error in this fixed pilot; generalization beyond this split remains unverified."
    )
    summary = {
        "status": "complete",
        "candidate": "coverage-matched reference agreement",
        "filtered_key_means": "Coverage in this experiment",
        "research_conclusion": conclusion,
        "comparison": comparison,
        "base": read_json(runs / "BASE.json"),
        "design": design,
        "data_policy": read_json(data / "POLICY.json"),
        "training": training,
        "filler_groups": read_json(runs / "FILLER_GROUPS.json"),
        "train_dev_features": read_json(runs / "TRAIN_DEV_FEATURES.json"),
        "local_verification": verification,
        "runtime_source_hashes": read_json(runs / "SOURCE.json"),
        "preparation_source_hashes": read_json(runs / "PREPARATION_SOURCE.json"),
        "reporting_source_hashes": {
            name: file_hash(Path(__file__).parent / name)
            for name in [
                "prepare_duplexlab_coverage.py",
                "verify_duplexlab_coverage.py",
                "report_duplexlab_coverage.py",
            ]
        },
        "limitations": [
            "Fixed selection seed; three training seeds do not measure dataset-selection variance.",
            "Test speakers appeared in previous training pool; current six SFTs all restart from base with these speakers excluded.",
            "Historical glossary extraction read all official train texts; this experiment uses no such glossary.",
            "Speaker-disjoint current roles; complete conversation-level independence is not verified.",
            "Difficulty is base/reference disagreement; filler presence is a lexical proxy, not human acoustic judgment.",
            "Only three test speakers and three training seeds; intervals are exploratory. Public-data pretraining exposure unknown.",
            "Matched marginals do not match every joint interaction or exact token/FLOP budget. Human reference verification pending.",
        ],
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    if any(s in text for s in ["/Users/", "/mnt/", '"reference"', '"hypothesis"']):
        raise ValueError("Private paths or full text leaked into public aggregate")
    (out / "SUMMARY.json").write_text(text + "\n", encoding="utf-8")
    write_json(out / "COVERAGE_AUDIT.json", design)
    write_json(out / "TRAIN_DEV_FEATURES.json", summary["train_dev_features"])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "svg.hashsalt": "duplexlab-coverage-v1",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = ["#316bd0", "#a266bd", "#12907b"]
    test = comparison["test"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for i, (seed, row) in enumerate(test["per_seed"].items()):
        axes[0].plot(
            ["Random", "Coverage"],
            [row["arms"][a]["mer"] * 100 for a in ["random", "filtered"]],
            marker="o",
            color=colors[i],
            label=f"Seed {seed}",
        )
        values = [row["delta"] * 100] + [
            (
                row["languages"][g]["filtered"]["mer"]
                - row["languages"][g]["random"]["mer"]
            )
            * 100
            for g in ["zh", "mixed", "en"]
        ]
        axes[1].bar(
            [v + (i - 1) * 0.24 for v in range(4)], values, width=0.22, color=colors[i]
        )
    axes[0].set(
        ylabel="MER (%)", title="New 300 clips / current split speaker-disjoint"
    )
    axes[0].legend(frameon=False)
    axes[1].axhline(0, color="#65758b", linewidth=1)
    axes[1].set_xticks(range(4), ["Overall", "Chinese", "Mixed", "English"])
    axes[1].set(
        ylabel="Coverage minus Random (percentage points)",
        title="Negative means fewer Coverage errors",
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    for suffix in ["png", "svg"]:
        fig.savefig(
            out / f"test-seeds.{suffix}",
            dpi=180,
            metadata={"Date": None} if suffix == "svg" else None,
        )
    plt.close(fig)
    counts = design["audit"]["marginals"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    bottom = [0, 0, 0]
    for level, color in zip(
        ["easy", "medium", "hard"], ["#91b8ee", "#447acb", "#193f79"]
    ):
        values = [
            counts.get(f"difficulty|{g}|{level}", 0) for g in ["zh", "mixed", "en"]
        ]
        axes[0].bar(
            ["Chinese", "Mixed", "English"],
            values,
            bottom=bottom,
            color=color,
            label=level,
        )
        bottom = [a + b for a, b in zip(bottom, values)]
    axes[0].set(title="Difficulty proxy counts", ylabel="Training clips")
    axes[0].legend(frameon=False, fontsize=8)
    speakers = sorted(
        {int(k.split("|")[-1]) for k in counts if k.startswith("speaker|")}
    )
    values = [
        sum(
            v
            for k, v in counts.items()
            if k.startswith("speaker|") and k.endswith("|" + str(s))
        )
        for s in speakers
    ]
    axes[1].bar([str(s) for s in speakers], values, color="#316bd0")
    axes[1].set(title="Speaker counts", xlabel="Official speaker ID")
    axes[2].bar(
        ["Chinese", "Mixed", "English"],
        [counts[f"filler|{g}|1"] for g in ["zh", "mixed", "en"]],
        color="#12907b",
    )
    axes[2].set(
        title="Filler-present clips", ylabel="Lexical proxy; human labels pending"
    )
    fig.suptitle(
        "Shared counts in Random500 and Coverage500; matched marginals, not all joint interactions"
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    for suffix in ["png", "svg"]:
        fig.savefig(
            out / f"coverage-design.{suffix}",
            dpi=180,
            metadata={"Date": None} if suffix == "svg" else None,
        )
    plt.close(fig)

    replay = runs / "replay"
    replay.mkdir(exist_ok=True)
    style = """<style>body{font:16px/1.6 system-ui;background:#f1f5f9;color:#172235;max-width:1100px;margin:auto;padding:24px}header,article{background:white;border:1px solid #d7e1eb;border-radius:12px;padding:20px;margin:16px 0}h1{font-size:28px}h2{font-size:17px}p{overflow-wrap:anywhere;white-space:pre-wrap}.models{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px}audio{max-width:100%}select{font:inherit;padding:6px}label{display:inline-block;margin:4px}article[data-category=worse]{border-left:4px solid #b75b66}article[data-category=better]{border-left:4px solid #12907b}@media(max-width:650px){body{padding:10px}.models{grid-template-columns:1fr}}</style>"""

    def panel(name, row):
        color = (
            "#12907b"
            if name == "Coverage"
            else "#316bd0"
            if name == "Random"
            else "#65758b"
        )
        return f'<section style="border-top:3px solid {color}"><b>{name} · S/D/I={row["S"]}/{row["D"]}/{row["I"]}</b><p>{escape(row["hypothesis"])}</p></section>'

    def errors(row):
        return row["S"] + row["D"] + row["I"]

    for role in ["dev", "test"]:
        source = rows(data / f"{role}.jsonl")
        scores = {
            f"{a}-{s}": {
                r["clip_id"]: r
                for r in read_json(runs / f"{role}-{a}-{s}/summary.json")["per_clip"]
            }
            for s in [42, 43, 44]
            for a in ["random", "filtered"]
        }
        base = {
            r["clip_id"]: r
            for r in read_json(runs / f"{role}-base/summary.json")["per_clip"]
        }
        html = (
            '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            + style
        )
        html += f"<title>Coverage {role} review</title><header><h1>Coverage · {role} {len(source)}</h1><p>Three fresh training seeds. Original source references retained; human listening pending. Coverage matches training marginals. Strict edit counts define filters.</p>"
        html += """<label>Seed <select id="seed"><option>42</option><option>43</option><option>44</option></select></label><label>Effect <select id="effect"><option value="all">All</option><option value="worse">Coverage: more edits</option><option value="better">Coverage: fewer edits</option><option value="equal">Equal edits</option></select></label><label>Language <select id="lang"><option value="all">All</option><option>zh</option><option>mixed</option><option>en</option></select></label><label>Filler proxy <select id="filler"><option value="all">All</option><option value="yes">Present</option><option value="no">Absent</option></select></label></header>"""
        for row in source:
            cid = row["clip_id"]
            filler = "yes" if has_filler(row["reference"]) else "no"
            html += f'<article data-language="{row["language_group"]}" data-filler="{filler}"><h2>{cid} · speaker {row["original_speaker_id"]} · {row["language_group"]} · filler {filler}</h2><audio controls preload="none" src="{(data / row["audio_relpath"]).resolve().as_uri()}"></audio><p>Source: {escape(row["reference"])}</p>'
            for seed in [42, 43, 44]:
                a, b = scores[f"random-{seed}"][cid], scores[f"filtered-{seed}"][cid]
                delta = errors(b) - errors(a)
                cat = "better" if delta < 0 else "worse" if delta > 0 else "equal"
                html += (
                    f'<div class="models seed-panel" data-seed="{seed}" data-category="{cat}">'
                    + panel("Base", base[cid])
                    + panel("Random", a)
                    + panel("Coverage", b)
                    + "</div>"
                )
            html += "</article>"
        html += """<script>function update(){const seed=document.querySelector('#seed').value,effect=document.querySelector('#effect').value,lang=document.querySelector('#lang').value,filler=document.querySelector('#filler').value;for(const a of document.querySelectorAll('article')){let cat;for(const p of a.querySelectorAll('.seed-panel')){const show=p.dataset.seed===seed;p.style.display=show?'':'none';if(show)cat=p.dataset.category}a.dataset.category=cat;a.hidden=(effect!=='all'&&effect!==cat)||(lang!=='all'&&lang!==a.dataset.language)||(filler!=='all'&&filler!==a.dataset.filler)}}for(const s of document.querySelectorAll('select'))s.onchange=update;update();</script></html>"""
        (replay / f"{role}.html").write_text(html, encoding="utf-8")
    write_json(
        runs / "REPORT_ARTIFACTS.json",
        {
            "summary_sha256": file_hash(out / "SUMMARY.json"),
            "private_test_cards": 300,
            "private_dev_cards": 40,
        },
    )
    print(out)


if __name__ == "__main__":
    main()
