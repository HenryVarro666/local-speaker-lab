"""Human listening packets and source-bound, append-only correction imports."""
from __future__ import annotations

from datetime import datetime
from html import escape
import json
from pathlib import Path

from .protocol import audio_path, content_hash, read_json, verify_protocol
from .report import CSS
from .store import RunStore


def build_review_packet(run_dir: Path, output_dir: Path) -> Path:
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    state = read_json(run_dir / "run.json")
    records = verify_protocol(Path(state["protocol_dir"]), Path(state["data_root"]))
    predictions = {r["clip_id"]: r for r in RunStore(run_dir / "state.sqlite").iter_results()}
    existing = RunStore(run_dir / "state.sqlite").iter_reviews()
    latest = {r["clip_id"]: content_hash(r) for r in existing}
    forms, drafts = [], []
    for row in records:
        cid = row["clip_id"]
        forms.append(f"<article data-clip='{escape(cid, quote=True)}'><h2>{escape(cid)}</h2>"
                     f"<audio controls preload='none' src='{escape(audio_path(row, Path(state['data_root'])).as_uri(), quote=True)}'></audio>"
                     "<label><input type='checkbox' data-field='listened'>我已回听这段音频</label>"
                     "<label>听到的文字 / 纠正后的转写<textarea data-field='corrected_text'></textarea></label>"
                     "<label>决策<select data-field='decision'><option value='defer'>待审阅</option>"
                     "<option value='approve'>确认转写</option><option value='reject'>排除</option></select></label>"
                     f"<details><summary>查看原参考与模型输出（本页不宣称盲审）</summary>"
                     f"<pre>参考：{escape(row['reference'])}\n预测：{escape(predictions.get(cid, {}).get('hypothesis', 'missing'))}</pre></details></article>")
        drafts.append({"schema_version": 1, "protocol_hash": state["protocol_hash"], "clip_id": cid,
                       "audio_sha256": row["audio_sha256"], "reviewer_id": "", "listened": False,
                       "corrected_text": "", "decision": "defer", "answer_exposed": True,
                       "supersedes": latest.get(cid), "reviewed_at": None})
    payload = json.dumps(drafts, ensure_ascii=False).replace("<", "\\u003c")
    javascript = """
const drafts=JSON.parse(document.getElementById('drafts').textContent);
document.getElementById('export').onclick=()=>{
 const reviewer=document.getElementById('reviewer').value.trim();
 if(!reviewer){alert('请先填写审阅者');return;}
 const completed=[];
 document.querySelectorAll('article').forEach((section,index)=>{
  const row={...drafts[index],reviewer_id:reviewer,reviewed_at:new Date().toISOString()};
  row.listened=section.querySelector('[data-field=listened]').checked;
  row.corrected_text=section.querySelector('[data-field=corrected_text]').value.trim();
  row.decision=section.querySelector('[data-field=decision]').value;
  if(row.decision!=='defer')completed.push(row);
 });
 if(completed.some(r=>!r.listened||(r.decision==='approve'&&!r.corrected_text))){alert('确认前请回听并填写实际转写');return;}
 const blob=new Blob([JSON.stringify(completed,null,2)],{type:'application/json'});
 const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='reviews.json';link.click();URL.revokeObjectURL(link.href);
};
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "review.html"
    output.write_text("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>DuplexLab listening review</title>"
                      f"<meta name='viewport' content='width=device-width,initial-scale=1'><style>{CSS}</style>"
                      "<h1>人工回听与转写复核</h1><p>此页目前待人工处理。页面生成不等于已回听；从结果进入按非盲审记录。</p>"
                      "<label>审阅者<input id='reviewer'></label><button id='export'>导出已处理条目</button>"
                      + "".join(forms) + f"<script type='application/json' id='drafts'>{payload}</script>"
                      + f"<script>{javascript}</script></html>", encoding="utf-8")
    return output


def import_reviews(run_dir: Path, review_file: Path) -> dict:
    run_dir = Path(run_dir)
    state = read_json(run_dir / "run.json")
    records = {r["clip_id"]: r for r in verify_protocol(Path(state["protocol_dir"]), Path(state["data_root"]))}
    entries = read_json(review_file)
    if not isinstance(entries, list):
        raise ValueError("Reviews must be a JSON array")
    store = RunStore(run_dir / "state.sqlite")
    existing = store.iter_reviews()
    known = {content_hash(r) for r in existing}
    latest = {r["clip_id"]: content_hash(r) for r in existing}
    seen, pending = set(), []
    for entry in entries:
        if not isinstance(entry, dict) or type(entry.get("schema_version")) is not int or entry["schema_version"] != 1:
            raise ValueError("Invalid review schema")
        cid = entry.get("clip_id")
        if cid not in records or cid in seen:
            raise ValueError("Unknown or duplicate review clip_id")
        seen.add(cid)
        if entry.get("audio_sha256") != records[cid]["audio_sha256"] or entry.get("protocol_hash") != state["protocol_hash"]:
            raise ValueError("Review audio/protocol hash mismatch")
        if not isinstance(entry.get("reviewer_id"), str) or not entry["reviewer_id"].strip():
            raise ValueError("Missing reviewer_id")
        if entry.get("decision") not in {"approve", "reject", "defer"}:
            raise ValueError("Invalid review decision")
        if type(entry.get("listened")) is not bool or type(entry.get("answer_exposed")) is not bool:
            raise ValueError("Review listening/exposure flags must be boolean")
        if entry["decision"] != "defer" and entry["listened"] is not True:
            raise ValueError("Human approval/rejection requires actual listening")
        if entry["decision"] == "approve" and (not isinstance(entry.get("corrected_text"), str) or not entry["corrected_text"].strip()):
            raise ValueError("Approval requires corrected_text")
        try:
            datetime.fromisoformat(entry["reviewed_at"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid reviewed_at") from error
        identity = content_hash(entry)
        if identity in known:
            continue
        if entry.get("supersedes") != latest.get(cid):
            raise ValueError("Conflicting review: explicitly supersede the current version")
        pending.append(entry)
    for entry in pending:
        store.put_review(entry)
    return {"inserted": len(pending), "duplicates": len(entries) - len(pending)}
