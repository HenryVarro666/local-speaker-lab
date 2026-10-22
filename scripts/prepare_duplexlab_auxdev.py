"""Freeze twelve new dev challenges by lexical coverage, never prediction error."""

import hashlib
import io
import json
import tempfile
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.optimize import Bounds, LinearConstraint, milp
from speaker_lab.duplex.audio import audit_clip
from speaker_lab.duplex.disfluency import text_features
from speaker_lab.duplex.datasets import validate_partitions
from speaker_lab.duplex.protocol import file_hash, write_json

ROOT = Path(__file__).resolve().parents[1]


def load(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def main():
    d = ROOT / "data/duplexlab/disfluency-v1"
    runs = ROOT / "runs/duplexlab/disfluency-v1"
    assert not (d / "AUXDEV_POLICY.json").exists()
    parts = {role: load(d / f"{role}.jsonl") for role in ["train", "dev", "test"]}
    prior = sum(parts.values(), [])
    for name in [
        "data/duplexlab/coverage-v1/test.jsonl",
        "data/duplexlab/multiseed-v1/primary.jsonl",
        "data/duplexlab/continuation-v1/train.jsonl",
        "data/duplexlab/continuation-v1/test.jsonl",
        "data/duplexlab/continuation-v1/language_confirmation.jsonl",
        "data/duplexlab/dev20-ascend-train-v1/manifest.jsonl",
    ]:
        prior += load(ROOT / name)
    ids = {r["clip_id"] for r in prior}
    sha = {r["audio_sha256"] for r in prior}
    pcm = {r["pcm_sha256"] for r in prior if r.get("pcm_sha256")}
    candidates = []
    waves = {}
    audits = []
    src = Path(
        "/Users/caochao/.cache/huggingface/hub/datasets--CAiRE--ASCEND/snapshots/737e9800ae31be9932ba8464c80366559bd28424/main"
    )
    with tempfile.TemporaryDirectory(dir=d) as tmp:
        temp = Path(tmp)
        (temp / "original").mkdir()
        for shard in sorted(src.glob("train-0000[12]-of-00003.parquet")):
            for source in pq.read_table(shard).to_pylist():
                if (
                    source["original_speaker_id"] not in [5, 6]
                    or not source["transcription"].strip()
                ):
                    continue
                cid = "ascend-train-" + source["id"]
                raw = source["audio"]["bytes"]
                info = sf.info(io.BytesIO(raw))
                if cid in ids or not 1 <= info.duration <= 12:
                    continue
                row = dict(
                    schema_version=1,
                    clip_id=cid,
                    audio_relpath=f"original/{cid}.wav",
                    audio_sha256=hashlib.sha256(raw).hexdigest(),
                    source_id="CAiRE/ASCEND@737e9800ae31be9932ba8464c80366559bd28424",
                    source_group_id=f"ascend-speaker-{source['original_speaker_id']}",
                    original_speaker_id=source["original_speaker_id"],
                    session_id=source["session_id"],
                    topic=source["topic"],
                    original_filename=Path(source["path"]).name,
                    source_shard=shard.name,
                    split="dev",
                    original_split="train",
                    language_group=source["language"],
                    reference=source["transcription"],
                    origin="real",
                    license_id="CC-BY-SA-4.0",
                    label_status="source_reference",
                    sample_rate=info.samplerate,
                    channels=info.channels,
                    frames=info.frames,
                    duration_seconds=info.duration,
                    split_unit="original_speaker_id",
                )
                path = temp / row["audio_relpath"]
                path.write_bytes(raw)
                a = audit_clip(row, temp, d / "auxdev-derived", {})
                audits.append(a)
                if (
                    a["decision"] == "pass"
                    and row["audio_sha256"] not in sha
                    and a["pcm_sha256"] not in pcm
                ):
                    row.update(
                        audit_decision="pass",
                        pcm_sha256=a["pcm_sha256"],
                        silence_fraction=a["silence_fraction"],
                    )
                    candidates.append(row)
                    waves[cid] = raw
                path.unlink()
    candidates.sort(
        key=lambda r: hashlib.sha256(
            ("20261022:auxdev:" + r["clip_id"]).encode()
        ).hexdigest()
    )
    features = [text_features(r["reference"]) for r in candidates]
    table = [[1] * len(candidates)]
    table += [[int(r["original_speaker_id"] == s) for r in candidates] for s in [5, 6]]
    table += [
        [int(r["language_group"] == lang) for r in candidates]
        for lang in ["zh", "en", "mixed"]
    ]
    table += [
        [int(f["filler_count"] >= 2) for f in features],
        [int(f["repeat_count"] >= 2) for f in features],
        [
            int(f["switch_count"] >= 4 and r["language_group"] == "mixed")
            for f, r in zip(features, candidates)
        ],
        [
            int(f["max_en_span"] >= 3 and r["language_group"] == "mixed")
            for f, r in zip(features, candidates)
        ],
    ]
    lower = [12, 4, 4, 2, 2, 6, 2, 2, 2, 2]
    upper = [12] + [np.inf] * 9
    solution = milp(
        np.arange(1, len(candidates) + 1, dtype=float),
        integrality=np.ones(len(candidates)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(np.array(table, dtype=float), lower, upper),
        options={"time_limit": 120, "mip_rel_gap": 0},
    )
    assert solution.status == 0 and solution.x is not None
    vector = np.rint(solution.x).astype(int)
    actual = np.array(table) @ vector
    assert all(actual >= lower) and actual[0] == 12
    chosen = sorted(
        [r for r, x in zip(candidates, vector) if x], key=lambda r: r["clip_id"]
    )
    assert len({r["pcm_sha256"] for r in chosen}) == 12
    validate_partitions(
        {"train": parts["train"], "dev": parts["dev"] + chosen, "test": parts["test"]}
    )
    for r in chosen:
        (d / r["audio_relpath"]).write_bytes(waves[r["clip_id"]])
    (d / "auxdev.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen)
    )
    write_json(d / "AUXDEV_AUDIT.json", audits)
    write_json(
        d / "AUXDEV_POLICY.json",
        {
            "count": 12,
            "speaker_ids": [5, 6],
            "constraint_names": [
                "count",
                "speaker5",
                "speaker6",
                "zh",
                "en",
                "mixed",
                "filler2+",
                "repeat2+",
                "mixed_switch4+",
                "mixed_en_span3+",
            ],
            "minimums": lower,
            "actual": actual.tolist(),
            "manifest_sha256": file_hash(d / "auxdev.jsonl"),
            "protocol_sha256": file_hash(
                ROOT / "experiments/duplexlab/DISFLUENCY_DEV_ADDENDUM.md"
            ),
            "new_test_unobserved_marker": json.loads(
                (runs / "TEST_UNOBSERVED_AUXDEV_FREEZE.json").read_text()
            ),
            "source_sha256": file_hash(Path(__file__)),
            "candidate_count": len(candidates),
            "solver_status": int(solution.status),
            "human_listening": "pending",
            "evaluation_after_main_pipeline": True,
        },
    )
    print(
        json.dumps(
            {
                "selected": 12,
                "constraints": actual.tolist(),
                "candidate_count": len(candidates),
            }
        )
    )


if __name__ == "__main__":
    main()
