"""Gradio front end. Model packages run only inside their worker environments."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def _read_text(path: str | None, fallback: str = "") -> str:
    return Path(path).read_text(encoding="utf-8") if path else fallback


def build_ui(config_path: str = "configs/default.toml"):
    import gradio as gr

    from .pipeline import run_pipeline

    def process(audio_path, reference_00, reference_01, with_summary, progress=gr.Progress()):
        # First yield clears previous outputs before this job starts.
        yield None, None, [], "", None, None, None, "准备处理…"
        if not audio_path:
            yield None, None, [], "", None, None, None, "请先上传一段双人录音。"
            return
        reference_paths = {
            key: path
            for key, path in (("SPEAKER_00", reference_00), ("SPEAKER_01", reference_01))
            if path
        }

        def stage(message: str):
            progress(0, desc=message)

        try:
            result = run_pipeline(
                audio_path=str(audio_path),
                config_path=config_path,
                reference_paths=reference_paths or None,
                progress=stage,
                summarize=bool(with_summary),
            )
            rows = [
                [
                    segment.get("id", ""),
                    round(float(segment["start"]), 2),
                    round(float(segment["end"]), 2),
                    segment.get("speaker", "UNKNOWN"),
                    segment.get("text_raw", segment.get("text", "")),
                    "是" if segment.get("uncertain") else "",
                ]
                for segment in result.get("segments", [])
            ]
            summary = _read_text(
                result.get("summary_path"),
                "本次未生成纪要。" if not with_summary else "纪要未生成，请查看运行状态。",
            )
            warnings = result.get("warnings", [])
            status = f"完成。结果目录：{result['run_dir']}"
            if warnings:
                status += "\n\n" + "\n".join(f"• {warning}" for warning in warnings)
            progress(1, desc="处理完成")
            yield (
                result["tracks"].get("SPEAKER_00"),
                result["tracks"].get("SPEAKER_01"),
                rows,
                summary,
                result.get("archive_path"),
                result.get("transcript_path"),
                result.get("subtitles_path"),
                status,
            )
        except Exception as exc:
            logger.exception("Audio pipeline failed")
            yield None, None, [], "", None, None, None, f"处理失败：{exc}\n请查看终端日志，并按安装文档检查模型环境、HF_TOKEN 和 Ollama 服务。"

    with gr.Blocks(title="Local Speaker Lab", analytics_enabled=False) as demo:
        gr.Markdown(
            "# Local Speaker Lab\n"
            "双人录音 → 人物音轨 → 带说话人的转写 → 本地会议纪要。\n\n"
            "人物编号按原录音首次发言顺序排列。上传后点击开始；模型首次下载可能较慢。"
        )
        audio = gr.Audio(label="原始双人录音", sources=["upload"], type="filepath")
        with gr.Accordion("可选：清晰参考语音（缺少单人发言片段时使用）", open=False):
            gr.Markdown("每段只包含对应人物的清晰声音。人物顺序以原录音首次发言为准；同时出现时请回听确认。")
            with gr.Row():
                reference_00 = gr.Audio(label="人物 1 参考（首次发言者，可选）", sources=["upload"], type="filepath")
                reference_01 = gr.Audio(label="人物 2 参考（第二位发言者，可选）", sources=["upload"], type="filepath")
        with_summary = gr.Checkbox(label="调用本地 Ollama 生成中文纪要", value=True)
        run_button = gr.Button("开始处理", variant="primary")
        status = gr.Textbox(label="运行状态与提示", value="等待上传录音。", interactive=False, lines=4)
        with gr.Row():
            track_00 = gr.Audio(label="人物 1 · SPEAKER_00", type="filepath", interactive=False)
            track_01 = gr.Audio(label="人物 2 · SPEAKER_01", type="filepath", interactive=False)
        gr.Markdown("## 带说话人的转写\n时间以原录音为基准；重叠说话允许时间范围重合。")
        transcript = gr.Dataframe(
            headers=["片段 ID", "开始（秒）", "结束（秒）", "人物", "原始转写", "待确认"],
            datatype=["str", "number", "number", "str", "str", "str"],
            interactive=False,
            wrap=True,
        )
        gr.Markdown("## 本地 LLM 纪要\n纪要由模型生成，请结合引用的片段和音频核对。")
        summary = gr.Markdown()
        with gr.Row():
            archive = gr.File(label="全部结果 ZIP", interactive=False)
            transcript_download = gr.File(label="转写 Markdown", interactive=False)
            subtitle_download = gr.File(label="带人物的字幕 SRT", interactive=False)
        run_button.click(
            fn=process,
            inputs=[audio, reference_00, reference_01, with_summary],
            outputs=[track_00, track_01, transcript, summary, archive, transcript_download, subtitle_download, status],
            concurrency_limit=1,
            concurrency_id="gpu-pipeline",
        )
    return demo.queue(default_concurrency_limit=1, max_size=8)


def main():
    parser = argparse.ArgumentParser(description="Local Speaker Lab Gradio UI")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    build_ui(args.config).launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
