"""Render aggregate figures and a private paired replay from completed runs."""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--language-comparison", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, destination = args.source, args.output
    destination.mkdir(parents=True, exist_ok=True)
    def load(path):
        return json.loads(path.read_text(encoding="utf-8"))
    records = {name: load(source / f"train-{name}-record/training.json") for name in ["smoke", "random", "filtered"]}
    if any(r["status"] != "trained_and_reloaded" or not r["checkpoint_reloaded"] for r in records.values()):
        raise ValueError("Training and real checkpoint reload must finish before final reporting")
    scores = {name: load(source / f"test-{name}/summary.json") for name in ["base", "random", "filtered"]}
    selection = load(source / "SELECTION.json")
    policy = load(args.data / "POLICY.json")
    language = load(args.language_comparison)
    comparison = load(source / "SFT_COMPARISON.json")
    direct = load(source / "SFT_FILTERED_VS_RANDOM.json")
    base_state = load(source / "test-base/run.json")
    environment = load(source / "environment.json")
    duration = selection["duration_seconds"]
    summary = {"schema_version": 1, "experiment_date": "2026-10-20", "status": "pilot_complete",
               "engineering_status": "offline_curation_sft_and_paired_evaluation_complete",
               "research_conclusion": "no clear evidence that reference-agreement filtering outperforms random; paired interval crosses zero",
               "language_confirmation": language,
               "selection": {"pool_clips": policy["counts"]["train"], "arms": selection["counts"],
                             "duration_seconds": duration, "duration_difference_fraction": abs(duration["random"] - duration["filtered"]) / duration["random"],
                             "overlap_clips": selection["overlap"], "duration_swaps": selection["audit"]["duration_swaps"], "quotas": policy["quotas"]},
               "training": {name: {k: r.get(k) for k in ["status", "optimizer_steps", "epoch", "losses", "subprocess_seconds", "checkpoint_reloaded", "export_hash", "upstream_sha256", "checkpoint_hashes"]} for name, r in records.items()},
               "test": comparison, "filtered_vs_random": direct,
               "test_languages": {name: score["origins"]["real"]["languages"] for name, score in scores.items()},
               "provenance": {"dataset_revision": policy["dataset_revision"], "parquet_sha256": policy["parquet_sha256"],
                              "manifest_sha256": policy["manifest_sha256"], "speaker_roles": policy["speakers"],
                              "base_model_id": "Qwen/Qwen3-ASR-0.6B", "base_model_revision": base_state["model_revision"],
                              "base_model_sha256": base_state["model_artifacts"],
                              "training_source_sha256": load(source / "TRAINING_SOURCE.json"),
                              "upstream_commit": "7c6daf77a2421100f5fb066495372c00129d39ff"},
               "environment": {k: environment[k] for k in ["python", "platform", "torch", "transformers", "qwen-asr", "datasets", "accelerate", "numpy", "soundfile", "pyarrow", "trainer_default_seed"]},
               "limitations": ["one training seed", "two test speakers; clip bootstrap is not speaker-cluster inference",
                               "public-data pretraining exposure unknown", "source_reference labels; no project human correction",
                               "offline only; no streaming, first-text, TTS, voice-to-voice or microphone measurement"]}
    if (source / "actual-training-arguments.json").exists():
        summary["actual_training_arguments"] = load(source / "actual-training-arguments.json")
    if (source / "export/EXPORT.json").exists():
        summary["provenance"]["training_export_files_sha256"] = load(source / "export/EXPORT.json")["files"]
    if (source / "gpu1-monitor-summary.json").exists():
        summary["gpu_memory"] = load(source / "gpu1-monitor-summary.json")
    text = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if '/Users/' in text or '/mnt/' in text or 'hypothesis' in text or 'reference"' in text:
        raise ValueError("Public aggregate unexpectedly contains private paths or transcripts")
    (destination / "SUMMARY.json").write_text(text, encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"svg.hashsalt": "duplexlab-continuation-v1", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    names, colors = ["base", "random", "filtered"], ["#65758b", "#3b70ce", "#109582"]
    labels = ["Overall MER", "Mixed MER", "Chinese CER", "English WER"]
    values = {name: [summary["test"]["runs"][name]["error_rate"]] + [summary["test_languages"][name][lang]["mer"] for lang in ["mixed", "zh", "en"]] for name in names}
    fig, axis = plt.subplots(figsize=(9, 4.3), layout="constrained")
    for i, name in enumerate(names):
        bars = axis.bar([x + (i - 1) * .25 for x in range(4)], values[name], width=.23, color=colors[i], label=name.title())
        axis.bar_label(bars, labels=[f"{v:.1%}" for v in values[name]], padding=3, fontsize=8)
    axis.set_xticks(range(4), labels)
    axis.set_ylim(0, max(v for vs in values.values() for v in vs) * 1.3 + .02)
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.set_title("Same frozen 100-clip test · one seed · two held-out speaker IDs", loc="left", pad=18)
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=.15)
    axis.set_axisbelow(True)
    fig.savefig(destination / "test-errors.svg", metadata={"Date": None})
    fig.savefig(destination / "test-errors.png", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 3.7), layout="constrained")
    for name, color in zip(names[1:], colors[1:]):
        loss = summary["training"][name]["losses"]
        axis.plot(range(1, len(loss) + 1), loss, label=name.title(), color=color, linewidth=1.5)
    axis.set(xlabel="Optimizer step", ylabel="Logged training loss", title="Training logs; downstream quality is measured on the separate frozen test")
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False)
    axis.grid(alpha=.15)
    fig.savefig(destination / "training-loss.svg", metadata={"Date": None})
    fig.savefig(destination / "training-loss.png", dpi=180)
    plt.close(fig)

    manifests = [json.loads(line) for line in (args.data / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {name: {r["clip_id"]: r for r in score["per_clip"]} for name, score in scores.items()}
    articles = []
    def edits(row):
        return row["S"] + row["D"] + row["I"]
    manifests.sort(key=lambda r: (-abs(edits(by_id['filtered'][r['clip_id']]) - edits(by_id['random'][r['clip_id']])), r['clip_id']))
    for row in manifests:
        cid = row["clip_id"]
        delta = edits(by_id["filtered"][cid]) - edits(by_id["random"][cid])
        category = "better" if delta < 0 else "worse" if delta > 0 else "equal"
        predictions = "".join(f"<section><h3>{name.title()} · edits {edits(by_id[name][cid])}</h3><p>{escape(by_id[name][cid]['hypothesis'])}</p></section>" for name in names)
        audio = (args.data / row["audio_relpath"]).resolve().as_uri()
        articles.append(f"<article data-category='{category}'><h2>{escape(cid)} · {escape(row['language_group'])} · speaker {row['original_speaker_id']}</h2>"
                        f"<audio controls preload='none' src='{escape(audio, quote=True)}'></audio><p><b>Source reference</b> {escape(row['reference'])}</p>"
                        f"<div class='predictions'>{predictions}</div></article>")
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DuplexLab checkpoint comparison</title>
<style>body{font:16px/1.55 system-ui;background:#f3f6fa;color:#172235;margin:0 auto;max-width:1160px;padding:24px}header,article{background:white;border:1px solid #dce3ed;border-radius:12px;padding:20px;margin:16px 0}h1{font-size:30px;margin:0}h2{font-size:17px}h3{font-size:14px}p{white-space:pre-wrap;overflow-wrap:anywhere}.predictions{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}section{border-top:3px solid #65758b}section:nth-child(2){border-color:#3b70ce}section:nth-child(3){border-color:#109582}audio{max-width:100%}select{font:inherit;padding:7px}article[data-category=worse]{border-left:4px solid #c15e64}article[data-category=better]{border-left:4px solid #109582}@media(max-width:650px){body{padding:12px}.predictions{grid-template-columns:1fr}h1{font-size:24px}}</style>
<header><h1>DuplexLab · checkpoint comparison</h1><p>100 frozen real recordings · automatic language · Base / Random-500 / Filtered-500. Original source references retained. This private replay has not been human reviewed. Filter uses strict edit-count differences between Filtered and Random.</p><label>Show <select id="filter"><option value="all">All clips</option><option value="better">Filtered: fewer edits</option><option value="worse">Filtered: more edits</option><option value="equal">Equal edit counts</option></select></label></header>'''
    html += "".join(articles) + '''<script>document.getElementById('filter').onchange=e=>{document.querySelectorAll('article').forEach(a=>{a.hidden=e.target.value!=='all'&&a.dataset.category!==e.target.value;});};</script></html>'''
    replay = source / "replay"
    replay.mkdir(exist_ok=True)
    (replay / "comparison.html").write_text(html, encoding="utf-8")
    print(json.dumps({"public_summary": str(destination / "SUMMARY.json"), "private_replay": str(replay / "comparison.html")}))


if __name__ == "__main__":
    main()
