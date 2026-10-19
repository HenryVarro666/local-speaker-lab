"""Fetch only text columns from pinned ASCEND train Parquet via HTTP ranges."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import time
import urllib.request


REPO = "CAiRE/ASCEND"
REVISION = "737e9800ae31be9932ba8464c80366559bd28424"


class HTTPRangeReader(io.BufferedIOBase):
    def __init__(self, url: str, size: int):
        self.url, self.size, self.position = url, size, 0
        self.downloaded_bytes = self.requests = 0
        self.cache = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        start = self.position
        size = min(self.size - start, self.size - start if size < 0 else size)
        if size <= 0:
            return b""
        if size > 8 * 1024 * 1024:
            raise ValueError("Refusing a large range: only transcript columns may be fetched")
        end = start + size - 1
        key = start, end
        if key not in self.cache:
            expected_range = f"bytes {start}-{end}/{self.size}"
            for attempt in range(3):
                try:
                    request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(request, timeout=45) as response:
                        if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                            raise OSError("Server did not honor the requested byte range")
                        data = response.read(size + 1)
                    if len(data) != size:
                        raise OSError("Truncated or oversized HTTP range")
                    break
                except OSError:
                    if attempt == 2:
                        raise
                    time.sleep(1 + attempt)
            self.cache[key] = data
            self.downloaded_bytes += len(data)
            self.requests += 1
        self.position += size
        return self.cache[key]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    api = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/main"
    with urllib.request.urlopen(api, timeout=30) as response:
        listing = json.load(response)
    sources = sorted([item for item in listing if re.fullmatch(r"main/train-\d{5}-of-\d{5}\.parquet", item["path"])],
                     key=lambda item: item["path"])
    if len(sources) != 3:
        raise ValueError("Expected all three pinned official training shards")
    output.parent.mkdir(parents=True, exist_ok=True)
    count, seen, provenance = 0, set(), []
    with tempfile.TemporaryDirectory(prefix=".ascend-train-text-", dir=output.parent) as temporary:
        staging = Path(temporary)
        text_path = staging / "train.jsonl"
        with text_path.open("x", encoding="utf-8", newline="\n") as stream:
            for item in sources:
                url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{item['path']}"
                with HTTPRangeReader(url, item["size"]) as reader:
                    parquet = pq.ParquetFile(reader)
                    shard_count = 0
                    for batch in parquet.iter_batches(columns=["id", "transcription", "language"], batch_size=1024):
                        for row in batch.to_pylist():
                            if row["id"] in seen or not all(isinstance(row[k], str) for k in row):
                                raise ValueError("Invalid or duplicate training row")
                            seen.add(row["id"])
                            record = {"id": row["id"], "reference": row["transcription"], "language": row["language"]}
                            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                            count += 1
                            shard_count += 1
                    if shard_count != parquet.metadata.num_rows:
                        raise ValueError("Training shard was not fully exported")
                    provenance.append({"path": item["path"], "size": item["size"],
                                       "lfs_sha256": item["lfs"]["oid"], "rows": shard_count,
                                       "transferred_bytes": reader.downloaded_bytes, "range_requests": reader.requests})
                    print(f"Exported train shard {len(provenance)}/3 ({shard_count} rows, {reader.downloaded_bytes} bytes)", flush=True)
        metadata = {"repo_id": REPO, "revision": REVISION, "split": "train", "license": "CC-BY-SA-4.0",
                    "columns": ["id", "transcription", "language"], "count": count, "sources": provenance,
                    "text_sha256": hashlib.sha256(text_path.read_bytes()).hexdigest(),
                    "export_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (staging / "dataset.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        staging.rename(output)
    print(f"Exported {count} training transcripts to {output}")


if __name__ == "__main__":
    main()
