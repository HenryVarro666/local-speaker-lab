"""Aggregate strict scores and lexical deletion diagnostics from completed runs."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from speaker_lab.duplex.disfluency import alignment_edits, text_features
from speaker_lab.duplex.protocol import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    runs = ROOT / "runs/duplexlab/disfluency-v1"
    data = ROOT / "data/duplexlab/disfluency-v1"
    result = {}
    for role in ["dev", "test", "auxdev"]:
        meta = {
            r["clip_id"]: r
            for r in map(json.loads, (data / f"{role}.jsonl").read_text().splitlines())
        }
        result[role] = {}
        for seed in [42, 43, 44]:
            result[role][str(seed)] = {}
            for arm in ["random", "filtered"]:
                scores = json.loads(
                    (runs / f"{role}-{arm}-{seed}/summary.json").read_text()
                )["per_clip"]
                bins = defaultdict(Counter)
                for row in scores:
                    ref = meta[row["clip_id"]]["reference"]
                    f = text_features(ref)
                    e = alignment_edits(ref, row["hypothesis"])
                    assert all(e[k] == row[k] for k in ["S", "D", "I", "N"])
                    count = {
                        k: e[k] for k in ["S", "D", "I", "N", "deleted_filler_count"]
                    }
                    count.update(clips=1, ref_filler_count=f["filler_count"])
                    single = sum(
                        len(x["token"]) == 1
                        and x["token"].isascii()
                        and x["token"].isalpha()
                        for x in e["deleted"]
                    )
                    count.update(
                        deleted_single_ascii_letter_count=single,
                        deleted_other_count=e["D"] - e["deleted_filler_count"] - single,
                    )

                    def b(n):
                        return str(n) if n < 2 else "2+"

                    sw = f["switch_count"]
                    en = f["max_en_span"]
                    groups = [
                        "overall",
                        f"frequency|{b(f['filler_count'])}",
                        f"repetition|{b(f['repeat_count'])}",
                        f"switches|{sw if sw < 2 else '2-3' if sw < 4 else '4+'}",
                        f"en_span|{'0' if en == 0 else '1-2' if en < 3 else '3+'}",
                    ]
                    for g in groups:
                        bins[g].update(count)
                values = {}
                for k, c in bins.items():
                    v = dict(c)
                    v["mer"] = (c["S"] + c["D"] + c["I"]) / c["N"] if c["N"] else None
                    v["filler_deletion_rate"] = (
                        c["deleted_filler_count"] / c["ref_filler_count"]
                        if c["ref_filler_count"]
                        else None
                    )
                    values[k] = v
                result[role][str(seed)][arm] = values
    write_json(runs / "DISFLUENCY_GROUPS.json", result)
    print(
        "All18 SFT evaluation batches grouped; strict counts and lexical deletion diagnostics verified"
    )


if __name__ == "__main__":
    main()
