"""Verified three-arm deletion experiment: public figures and private playback."""

from __future__ import annotations
import json
from html import escape
from pathlib import Path
from speaker_lab.duplex.protocol import read_json, write_json, file_hash
from speaker_lab.duplex.disfluency import text_features

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["random", "selected", "filtered"]
LABELS = ["Random CE", "Selected CE", "Selected retention CE"]


def load(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    runs = ROOT / "runs/duplexlab/deletion-v1"
    data = ROOT / "data/duplexlab/deletion-v1"
    out = ROOT / "experiments/duplexlab/reports/2026-10-20-deletion"
    out.mkdir(parents=True, exist_ok=True)
    verification = read_json(runs / "LOCAL_VERIFICATION.json")
    assert verification["status"] == "verified"
    selection = read_json(runs / "SELECTION.json")
    groups = read_json(runs / "DISFLUENCY_GROUPS.json")
    comparisons = {
        role: {
            c: read_json(runs / f"COMPARISON-{role}-{c}.json")
            for c in ["joint", "selection", "objective"]
        }
        for role in ["dev", "test"]
    }
    train = {}
    for seed in [42, 43, 44]:
        for arm in ARMS:
            r = read_json(runs / f"train-{arm}-{seed}-record/training.json")
            assert r["status"] == "trained_and_reloaded" and r["test_complete"]
            train[f"{arm}-{seed}"] = {
                k: r[k]
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
            if arm == "filtered":
                train[f"{arm}-{seed}"]["loss_runtime"] = r["loss_runtime"]
    counts = {
        a: {
            k: sum(groups["test"][str(s)][a]["overall"][k] for s in [42, 43, 44])
            for k in [
                "S",
                "D",
                "I",
                "N",
                "deleted_filler_count",
                "deleted_single_ascii_letter_count",
                "deleted_other_count",
            ]
        }
        for a in ARMS
    }
    frozen = read_json(data / "POLICY.json")
    summary = dict(
        status="complete",
        arms=dict(zip(ARMS, LABELS)),
        counts=frozen["counts"],
        test_speakers=frozen["test_speakers"],
        comparison=comparisons,
        three_repeat_totals=counts,
        groups=groups,
        deletion_comparison=read_json(runs / "DELETION_COMPARISONS.json"),
        stable_deletions=read_json(runs / "STABLE_DELETIONS.json"),
        new_stable_extra=read_json(runs / "NEW_STABLE_EXTRA.json"),
        base=read_json(runs / "BASE.json"),
        training=train,
        selection_audit=selection["audit"],
        duration_seconds=selection["duration_seconds"],
        prior_analysis=read_json(runs / "PRIOR_ANALYSIS.json"),
        prior_analysis_source_hashes=read_json(runs / "PRIOR_ANALYSIS_SOURCE.json"),
        verification=verification,
        freeze=dict(
            protocol_sha256=frozen["protocol_sha256"],
            manifest_sha256=frozen["manifest_sha256"],
            selection_sha256=frozen["selection_sha256"],
            loss_weights_sha256=frozen["loss_weights_sha256"],
            execution_addendum_sha256=frozen["execution_addendum_sha256"],
        ),
        runtime_source_hashes=read_json(runs / "SOURCE.json"),
        preparation_source_hashes=read_json(runs / "PREPARATION_SOURCE.json"),
        reporting_source_hashes={
            f: file_hash(ROOT / "scripts" / f)
            for f in [
                "prepare_duplexlab_deletion.py",
                "run_duplexlab_deletion.py",
                "verify_duplexlab_deletion.py",
                "report_duplexlab_deletion.py",
                "summarize_duplexlab_deletion.py",
                "analyze_duplexlab_deletions.py",
            ]
        },
        reference_tokens_per_training_arm={
            a: sum(
                len(
                    __import__(
                        "speaker_lab.glossary_metrics", fromlist=["normalize_tokens"]
                    ).normalize_tokens(r["reference"])
                )
                for r in selection["arms"][a]
            )
            for a in ARMS
        },
        execution_attempts=dict(
            final_fresh_complete=9,
            excluded_attempt0_complete=2,
            excluded_attempt0_partial=1,
            failed_incomplete_transfer_preflight=1,
            eval_batch_size=1,
        ),
        limitations=[
            "New recordings, seven current-role-separated speaker IDs with historical project exposure; historical train-text glossary exposure disclosed.",
            "Fixed selection seed, three training seeds, exploratory whole-speaker intervals; no population guarantee or multiplicity adjustment.",
            "Source references retained; deletion alignment and lexical proxies do not establish acoustic ground truth. Human listening pending.",
            "Same examples/steps/hyperparameters and <=1% audio gap, not exact BPE tokens/FLOPs. Dev52 are previously observed known development examples.",
        ],
    )
    payload = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    assert not any(
        x in payload for x in ["/Users/", "/mnt/", '"reference"', '"hypothesis"']
    )
    (out / "SUMMARY.json").write_text(payload + "\n")
    write_json(out / "PRIOR_ANALYSIS.json", summary["prior_analysis"])
    write_json(
        out / "COVERAGE_AUDIT.json",
        {
            k: selection[k]
            for k in [
                "audit",
                "duration_seconds",
                "quotas",
                "pool_marginals",
                "source_predictions_sha256",
                "scipy",
            ]
        },
    )
    write_json(out / "DELETION_GROUPS.json", groups)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "svg.hashsalt": "duplexlab-deletion-v1",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = ["#316bd0", "#a266bd", "#12907b"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for j, seed in enumerate([42, 43, 44]):
        axes[0].plot(
            range(3),
            [groups["test"][str(seed)][a]["overall"]["mer"] * 100 for a in ARMS],
            marker="o",
            label=f"Seed {seed}",
            color=colors[j],
        )
        axes[1].plot(
            range(3),
            [
                groups["test"][str(seed)][a]["overall"]["D"]
                / groups["test"][str(seed)][a]["overall"]["N"]
                * 100
                for a in ARMS
            ],
            marker="o",
            color=colors[j],
        )
    for ax, title, ylabel in zip(
        axes,
        [
            "Strict MER on new 280 clips / 7 speakers",
            "Strict deletion rate on identical new test",
        ],
        ["MER (%)", "D / N (%)"],
    ):
        ax.set_xticks(range(3), ["Random CE", "Selected CE", "Retention CE"])
        ax.set(title=title, ylabel=ylabel)
        ax.grid(axis="y", alpha=0.15)
    axes[0].legend(frameon=False)
    for suffix in ["png", "svg"]:
        fig.savefig(
            out / f"test-seeds.{suffix}",
            dpi=180,
            metadata={"Date": None} if suffix == "svg" else None,
        )
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for j, a in enumerate(ARMS):
        axes[0].bar(
            [i + (j - 1) * 0.24 for i in range(3)],
            [counts[a][k] for k in ["S", "D", "I"]],
            width=0.22,
            color=colors[j],
            label=LABELS[j],
        )
    axes[0].set_xticks(range(3), ["Substitutions", "Deletions", "Insertions"])
    axes[0].set(
        title="Three repeats (not 840 independent recordings)", ylabel="Strict errors"
    )
    axes[0].legend(frameon=False, fontsize=8)
    for j, c in enumerate(["joint", "selection", "objective"]):
        x = comparisons["test"][c]
        lo, hi = x["hierarchical_delta_interval"]
        v = x["mean_delta"]
        axes[1].errorbar(
            v * 100,
            j,
            xerr=[[(v - lo) * 100], [(hi - v) * 100]],
            fmt="o",
            color=colors[j],
            capsize=4,
        )
    axes[1].set_yticks(
        range(3), ["Retention - Random", "Selected - Random", "Retention - Selected"]
    )
    axes[1].axvline(0, color="#65758b", lw=1)
    axes[1].set(
        title="Paired seed / whole-speaker 95% intervals",
        xlabel="MER difference (percentage points)",
    )
    for suffix in ["png", "svg"]:
        fig.savefig(
            out / f"deletion-contrasts.{suffix}",
            dpi=180,
            metadata={"Date": None} if suffix == "svg" else None,
        )
    plt.close(fig)
    replay = runs / "replay"
    replay.mkdir(exist_ok=True)
    style = "<style>body{font:16px/1.6 system-ui;background:#f1f5f9;color:#172235;max-width:1180px;margin:auto;padding:24px}header,article{background:white;border:1px solid #d7e1eb;border-radius:12px;padding:20px;margin:16px 0}h1{font-size:28px}h2{font-size:17px}p{overflow-wrap:anywhere;white-space:pre-wrap}.models{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px}audio{max-width:100%}select{font:inherit;padding:6px}label{display:inline-block;margin:4px}article[data-category=worse]{border-left:4px solid #b75b66}article[data-category=better]{border-left:4px solid #12907b}@media(max-width:700px){body{padding:10px}.models{grid-template-columns:1fr}}</style>"
    artifacts = {}
    for role in ["test", "dev"]:
        rows = load(data / f"{role}.jsonl")
        scores = {
            f"{a}-{s}": {
                r["clip_id"]: r
                for r in read_json(runs / f"{role}-{a}-{s}/summary.json")["per_clip"]
            }
            for s in [42, 43, 44]
            for a in ARMS
        }
        base = {
            r["clip_id"]: r
            for r in read_json(runs / f"{role}-base/summary.json")["per_clip"]
        }
        html = (
            '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            + style
            + f"<title>Deletion {role}</title><header><h1>Deletion retention · {role} {len(rows)}</h1><p>Three arms / three fresh seeds. Original references; human listening pending. Filters compare retention CE against Random CE. New recordings; speaker IDs have historical exposure.</p>"
        )
        html += '<label>Seed <select id="seed"><option>42</option><option>43</option><option>44</option></select></label><label>Metric <select id="metric"><option value="edits">All edits</option><option value="deletion">Deletions</option></select></label><label>Effect <select id="effect"><option value="all">All</option><option value="better">Fewer</option><option value="worse">More</option><option value="equal">Equal</option></select></label><label>Language <select id="lang"><option value="all">All</option><option>zh</option><option>mixed</option><option>en</option></select></label><label>Filler <select id="filler"><option value="all">All</option><option value="yes">Present</option><option value="no">Absent</option></select></label></header>'
        for row in rows:
            cid = row["clip_id"]
            f = text_features(row["reference"])
            filler = "yes" if f["filler_count"] else "no"
            html += f'<article data-id="{cid}" data-language="{row["language_group"]}" data-filler="{filler}"><h2>{cid} · speaker {row["original_speaker_id"]} · {row["language_group"]} · {row.get("dev_panel", "new test")}</h2><p>Fillers {f["filler_count"]} · repetition {f["repeat_count"]} · switches {f["switch_count"]}</p><audio controls preload="none" src="{(data / row["audio_relpath"]).resolve().as_uri()}"></audio><p>Source: {escape(row["reference"])}</p>'
            for seed in [42, 43, 44]:
                a = scores[f"random-{seed}"][cid]
                b = scores[f"filtered-{seed}"][cid]
                delta = sum(b[k] - a[k] for k in ["S", "D", "I"])
                dd = b["D"] - a["D"]

                def cat(x):
                    return "better" if x < 0 else "worse" if x > 0 else "equal"

                html += f'<div class="models seed-panel" data-seed="{seed}" data-edits="{cat(delta)}" data-deletion="{cat(dd)}">'
                for label, value in [("Base", base[cid])] + [
                    (LABELS[i], scores[f"{a}-{seed}"][cid]) for i, a in enumerate(ARMS)
                ]:
                    html += f"<section><b>{label} · S/D/I={value['S']}/{value['D']}/{value['I']}</b><p>{escape(value['hypothesis'])}</p></section>"
                html += "</div>"
            html += "</article>"
        html += """<script>function update(){const seed=document.querySelector('#seed').value,metric=document.querySelector('#metric').value,effect=document.querySelector('#effect').value,lang=document.querySelector('#lang').value,filler=document.querySelector('#filler').value;for(const a of document.querySelectorAll('article')){let cat;for(const p of a.querySelectorAll('.seed-panel')){const show=p.dataset.seed===seed;p.style.display=show?'':'none';if(show)cat=p.dataset[metric]}a.dataset.category=cat;a.hidden=(effect!=='all'&&cat!==effect)||(lang!=='all'&&lang!==a.dataset.language)||(filler!=='all'&&filler!==a.dataset.filler)}}for(const s of document.querySelectorAll('select'))s.onchange=update;update();</script></html>"""
        (replay / f"{role}.html").write_text(html)
        artifacts[role + "_cards"] = len(rows)
    prior = read_json(runs / "PRIOR_DELETIONS_PRIVATE.json")
    notes = {r["clip_id"]: r for r in summary["prior_analysis"]["cases"]}
    old_data = ROOT / "data/duplexlab/disfluency-v1"
    old_meta = {r["clip_id"]: r for r in load(old_data / "test.jsonl")}
    html = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        + style
        + "<style>.models{grid-template-columns:repeat(2,minmax(0,1fr))}@media(max-width:700px){.models{grid-template-columns:1fr}}</style><title>Prior stable deletion review</title><header><h1>Prior stable deletion review · 11 recordings</h1><p>Previously observed disfluency-v1 test. All three seeds and source references retained. Word position and acoustic truth are separate; human listening pending.</p></header>"
    )
    for row in prior:
        cid = row["clip_id"]
        if cid not in notes:
            continue
        note = notes[cid]
        html += f'<article data-id="{cid}"><h2>{cid}</h2><p>{escape(note["interpretation"])}</p><audio controls preload="none" src="{(old_data / old_meta[cid]["audio_relpath"]).resolve().as_uri()}"></audio><p>Source: {escape(row["reference"])}</p>'
        for token in row["stable_extra"]:
            html += f"<p>Reference position {token['index']}: {escape(token['token'])} · candidate forced in every optimal path: {token['candidate_forced_all_optimal_paths']} · random deleted seeds: {token['random_deleted_seeds']}</p>"
        for seed in [42, 43, 44]:
            html += f'<h3>Seed {seed}</h3><div class="models" data-seed="{seed}">'
            for arm, label in [
                ("random", "Prior Random CE"),
                ("filtered", "Prior Disfluency CE"),
            ]:
                value = row["alignment"][arm][str(seed)]
                html += f"<section><b>{label} · S/D/I={value['S']}/{value['D']}/{value['I']}</b><p>{escape(row['hypotheses'][arm][str(seed)])}</p></section>"
            html += "</div>"
        html += "</article>"
    html += "</html>"
    (replay / "prior.html").write_text(html)
    artifacts["prior_cards"] = len(notes)
    write_json(
        runs / "REPORT_ARTIFACTS.json",
        dict(artifacts, summary_sha256=file_hash(out / "SUMMARY.json")),
    )
    print(out)


if __name__ == "__main__":
    main()
