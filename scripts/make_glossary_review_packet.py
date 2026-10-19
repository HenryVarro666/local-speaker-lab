#!/usr/bin/env python3
"""Build a local-only listening worksheet without deciding what an audio clip says.

This deliberately reads only six diagnostic events from an already completed run.
Audio remains in its original location; file URLs are meaningful only locally.
"""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import sys


EVENT_IDS = ("00621", "00062", "00957", "01010", "00928", "00904")
CONDITIONS = ("none", "personal", "unrelated")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected a JSON object")
                rows.append(row)
    return rows


def script_json(value: object) -> str:
    """Escape script delimiters, including a malicious literal </script>."""
    return (json.dumps(value, ensure_ascii=False, allow_nan=False)
            .replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def build_packet(manifest: Path, run_dir: Path, output_dir: Path) -> Path:
    manifest, run_dir, output_dir = manifest.resolve(), run_dir.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    prediction_file, run_file = run_dir / "predictions.jsonl", run_dir / "run.json"
    provenance = json.loads(run_file.read_text(encoding="utf-8-sig"))
    if not isinstance(provenance, dict) or provenance.get("status") != "complete":
        raise ValueError("The source inference run must have status complete")
    manifest_sha = file_hash(manifest)
    if provenance.get("manifest_sha256") != manifest_sha:
        raise ValueError("Source run manifest SHA-256 does not match the supplied manifest")
    recorded_audio_hashes = provenance.get("audio_sha256", {})
    if not isinstance(recorded_audio_hashes, dict):
        raise ValueError("Source run audio_sha256 must be a JSON object")
    rows = {}
    for row in read_jsonl(manifest):
        clip_id = row.get("id")
        if clip_id in EVENT_IDS:
            if clip_id in rows:
                raise ValueError(f"Duplicate manifest event: {clip_id}")
            rows[clip_id] = row
    if set(rows) != set(EVENT_IDS):
        raise ValueError(f"Missing diagnostic event IDs: {sorted(set(EVENT_IDS) - set(rows))}")
    predictions = {}
    for row in read_jsonl(prediction_file):
        clip_id = row.get("id")
        if not isinstance(clip_id, str):
            raise ValueError("Prediction id must be a string")
        if clip_id not in rows:
            continue
        condition = row.get("condition")
        if condition not in CONDITIONS or not isinstance(row.get("text"), str):
            raise ValueError(f"Invalid diagnostic prediction: {clip_id}/{condition}")
        key = clip_id, condition
        if key in predictions:
            raise ValueError(f"Duplicate diagnostic prediction: {clip_id}/{condition}")
        predictions[key] = row
    expected = {(clip_id, arm) for clip_id in EVENT_IDS for arm in CONDITIONS}
    if set(predictions) != expected:
        raise ValueError(f"Incomplete diagnostic predictions: missing {len(expected - set(predictions))}")
    events = []
    # Stable ordering independent of the apparent error category or prediction outcome.
    for clip_id in sorted(EVENT_IDS, key=lambda item: hashlib.sha256(item.encode()).hexdigest()):
        row = rows[clip_id]
        if not isinstance(row.get("audio"), str) or not row["audio"].strip():
            raise ValueError(f"Missing local audio path: {clip_id}")
        if not isinstance(row.get("reference"), str):
            raise ValueError(f"Missing reference text: {clip_id}")
        audio_path = (manifest.parent / row["audio"]).resolve()
        if not audio_path.is_file():
            raise ValueError(f"Audio is not an existing local file: {audio_path}")
        audio_sha = file_hash(audio_path)
        recorded_sha = recorded_audio_hashes.get(clip_id)
        if recorded_sha is not None and recorded_sha != audio_sha:
            raise ValueError(f"Audio SHA-256 differs from the completed run: {clip_id}")
        events.append({
            "id": clip_id, "audio_path": str(audio_path), "audio_uri": audio_path.as_uri(),
            "audio_sha256": audio_sha, "reference": row["reference"],
            "predictions": {arm: predictions[clip_id, arm] for arm in CONDITIONS},
            "listened": False, "heard_text": None, "confidence": "not_reviewed",
            "prediction_exposure_before_listening": "unknown", "listened_at": None,
            "auditory_conclusion": None, "notes": None,
        })
    packet = {
        "schema_version": 1, "status": "pending_human_listening", "reviewer": None,
        "notice": "The automated agent cannot listen to audio. No human listening or acoustic "
                  "adjudication has been completed by generating this packet. Prior knowledge "
                  "of predictions means a review must not be called blind.",
        "policy": "Supplementary review only. Do not change the frozen vocabulary, references, "
                  "primary normalization, or primary scores based on these notes.",
        "license": "ASCEND recordings and reference annotations retain CC BY-SA 4.0. "
                   "No audio is copied by this tool. Keep this packet local; it includes local "
                   "paths and reference/prediction excerpts.",
        "dataset": {"name": "ASCEND", "split": "validation",
                    "source": "https://huggingface.co/datasets/CAiRE/ASCEND",
                    "license_url": "https://creativecommons.org/licenses/by-sa/4.0/"},
        "source": {"manifest": str(manifest), "manifest_sha256": manifest_sha,
                   "run_json_sha256": file_hash(run_file),
                   "predictions_sha256": file_hash(prediction_file),
                   "model_config": provenance.get("config"),
                   "seed": provenance.get("seed"), "versions": provenance.get("versions"),
                   "normalization": provenance.get("normalization"),
                   "personal_sha256": provenance.get("personal_sha256"),
                   "unrelated_sha256": provenance.get("unrelated_sha256")},
        "events": events,
    }
    cards = []
    for event in events:
        predictions_html = "".join(
            f"<dt>{escape(arm)}</dt><dd>{escape(event['predictions'][arm]['text'])}</dd>"
            for arm in CONDITIONS)
        cards.append(f"""<section data-id="{escape(event['id'], quote=True)}">
<h2>Event {escape(event['id'])}</h2>
<audio controls preload="none" src="{escape(event['audio_uri'], quote=True)}"></audio>
<p><a href="{escape(event['audio_uri'], quote=True)}">Open original local recording</a></p>
<label><input type="checkbox" data-field="listened"> I actually listened to this recording</label>
<label>What I heard (leave empty until listening)<textarea data-field="heard_text" rows="3"></textarea></label>
<label>Confidence<select data-field="confidence"><option value="not_reviewed">Not reviewed</option>
<option value="uncertain">Uncertain / unintelligible</option><option value="moderate">Moderate</option>
<option value="high">High</option></select></label>
<label>Had I seen a reference or prediction before listening?
<select data-field="prediction_exposure_before_listening"><option value="unknown">Unknown</option>
<option value="yes">Yes (not blind)</option><option value="no">No</option></select></label>
<label>Supplementary auditory conclusion<textarea data-field="auditory_conclusion" rows="2"></textarea></label>
<label>Notes, uncertainty, replay times<textarea data-field="notes" rows="2"></textarea></label>
<details><summary>Reference and model predictions — preferably open after listening</summary>
<p>Opening this panel is recorded. A previously informed reviewer is not blind even if this panel stays closed.</p>
<dl><dt>Reference</dt><dd>{escape(event['reference'])}</dd>{predictions_html}</dl></details>
</section>""")
    html = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ASCEND six-event local listening worksheet</title>
<style>body{max-width:900px;margin:2rem auto;padding:0 1rem;font:16px/1.5 system-ui}
section{border:1px solid #888;border-radius:8px;padding:1rem;margin:1.5rem 0}
label{display:block;margin:.7rem 0}textarea{display:block;width:95%;font:inherit}
audio{width:100%}dd{white-space:pre-wrap;overflow-wrap:anywhere}button{padding:.7rem}
.notice{padding:1rem;background:#fff1d6;color:#302000}</style>
<h1>ASCEND six-event local listening worksheet</h1>
<p class="notice">The automated agent cannot listen to audio. Generating this page does not complete
human listening or acoustic adjudication. These events are already prediction-aware diagnostics;
do not claim a blind review when you have previously seen their answers.</p>
<p>Keep this page and its downloaded JSON local. ASCEND audio/reference material retains
<a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>.
Audio is played from its original file URL and is not copied. If your browser blocks local audio,
open the recording link in a local player. Frozen primary scores remain unchanged.</p>
<p>Inputs are held in this page only; closing or reloading loses edits. Download the JSON to save
your work. Its review state remains pending until all six recordings have been listened to and annotated.</p>
<label>Reviewer name or identifier<input id="reviewer" autocomplete="off"></label>
""" + "\n".join(cards) + """
<button id="download" type="button">Download review.json</button><p id="message" role="status"></p>
<script type="application/json" id="packet-data">""" + script_json(packet) + """</script>
<script>
"use strict";
const packet = JSON.parse(document.getElementById("packet-data").textContent);
for (const section of document.querySelectorAll("section[data-id]")) {
  const item = packet.events.find(event => event.id === section.dataset.id);
  section.querySelector("details").addEventListener("toggle", event => {
    if (event.target.open) {
      item.reference_panel_opened_at = item.reference_panel_opened_at || new Date().toISOString();
      if (!section.querySelector('[data-field="listened"]').checked) {
        section.querySelector('[data-field="prediction_exposure_before_listening"]').value = "yes";
      }
    }
  });
}
document.getElementById("download").addEventListener("click", () => {
  packet.reviewer = document.getElementById("reviewer").value.trim() || null;
  for (const section of document.querySelectorAll("section[data-id]")) {
    const item = packet.events.find(event => event.id === section.dataset.id);
    for (const control of section.querySelectorAll("[data-field]")) {
      item[control.dataset.field] = control.type === "checkbox" ? control.checked : (control.value.trim() || null);
    }
    if (item.listened && !item.listened_at) item.listened_at = new Date().toISOString();
    if (!item.listened) item.listened_at = null;
  }
  packet.exported_at = new Date().toISOString();
  packet.status = packet.reviewer && packet.events.every(item => item.listened && item.heard_text &&
    item.confidence !== "not_reviewed" && item.prediction_exposure_before_listening !== "unknown")
    ? "reviewer_attested_listening_complete" : "pending_human_listening";
  const url = URL.createObjectURL(new Blob([JSON.stringify(packet, null, 2) + "\\n"], {type:"application/json"}));
  const link = document.createElement("a"); link.href = url; link.download = "review.json";
  document.body.appendChild(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  document.getElementById("message").textContent = "Downloaded review.json; status: " + packet.status;
});
</script></html>
"""
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "review_template.json").write_text(
        json.dumps(packet, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    destination = output_dir / "index.html"
    destination.write_text(html, encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(build_packet(args.manifest, args.run_dir, args.output_dir))
    except (OSError, ValueError, TypeError) as exc:
        print(f"Listening packet failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
