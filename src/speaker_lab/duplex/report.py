"""Private local audio replay and text-free public aggregate reports."""
from __future__ import annotations

from html import escape
from pathlib import Path

from .metrics import score_run
from .protocol import audio_path, read_json, verify_protocol


CSS = """
:root{color-scheme:light dark;font-family:system-ui,sans-serif}body{max-width:1080px;margin:40px auto;padding:0 22px}
h1{font-size:2rem}small,.note{opacity:.75}table{width:100%;border-collapse:collapse;margin:24px 0}
th,td{text-align:left;padding:10px;border-bottom:1px solid #8886}article{border:1px solid #8886;border-radius:12px;padding:20px;margin:16px 0}
audio{width:100%;margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}label{display:block;margin:10px 0}textarea{width:95%;min-height:70px}
button,select,input{font:inherit;padding:7px}a{color:light-dark(#145ba8,#8bbcff)}.tag{font-size:.8rem;border:1px solid #888;border-radius:5px;padding:3px 7px}
"""


def render_report(run_dir: Path, output: Path, public: bool = False) -> Path:
    run_dir, output = Path(run_dir), Path(output)
    state = read_json(run_dir / "run.json")
    summary = score_run(run_dir, Path(state["protocol_dir"]), Path(state["data_root"]))
    records = {r["clip_id"]: r for r in verify_protocol(Path(state["protocol_dir"]), Path(state["data_root"]))}
    rows = []
    for origin, group in summary["origins"].items():
        for language, result in [("all", group["overall"]), *group["languages"].items()]:
            rate = "undefined" if result["mer"] is None else f"{100 * result['mer']:.2f}%"
            rows.append(f"<tr><td>{escape(origin)}</td><td>{escape(language)}</td><td>{result['clips']}</td>"
                        f"<td>{result['S'] + result['D'] + result['I']} / {result['N']}</td><td>{rate}</td></tr>")
    content = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'>",
               "<meta name='viewport' content='width=device-width,initial-scale=1'>",
               f"<title>DuplexLab — ASR development replay</title><style>{CSS}</style>",
               "<h1>DuplexLab · 语音实验回放</h1>",
               "<p>真人语音 / 离线识别 / 开发集。未微调；没有原生全双工、真人麦克风或 TTS 首声测量。</p>",
               "<p class='note'>zh 为 CER、en 为 WER、mixed 为中英混合 MER；总分按总编辑数聚合。不是官方 ASCEND recipe。</p>",
               "<table><tr><th>来源</th><th>语言</th><th>录音数</th><th>编辑数 / 参考 token</th><th>错误率</th></tr>",
               *rows, "</table>"]
    if not public:
        content.append("<p><a href='review.html'>打开本地回听审阅页</a> · 原参考、预测与音频仅保留本机。</p>")
        for row in sorted(summary["per_clip"], key=lambda r: (-(r['S'] + r['D'] + r['I']), r['clip_id'])):
            path = audio_path(records[row["clip_id"]], Path(state["data_root"]))
            content.append(f"<article><h2>{escape(row['clip_id'])} <span class='tag'>{escape(row['language_group'])}</span></h2>"
                           f"<audio controls preload='none' src='{escape(path.as_uri(), quote=True)}'></audio>"
                           f"<p>原参考</p><pre>{escape(row['reference'])}</pre>"
                           f"<p>模型原始输出</p><pre>{escape(row['hypothesis'])}</pre>"
                           f"<p>S={row['S']} D={row['D']} I={row['I']} N={row['N']}</p></article>")
    else:
        content.append("<p>公开汇总：不包含录音、原文、预测或本机路径。</p>")
    content.append("</html>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(content), encoding="utf-8")
    return output
