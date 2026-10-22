# PersonaPlex scheduling Demo verification

2026-10-20, America/Chicago. The deployed Demo restores a clean voice/persona prefix inside one WebSocket, then prefills only current tool facts. **Complete authoritative tool wording is controlled; all audio is generated natively by PersonaPlex.** Spoken B K number one maps to displayed BK-0001. Device voice remains the labeled failure fallback; no model fine-tuning.

[Current latency report](native-injection/LATENCY_RESULTS.md) · [Numeric summary](SUMMARY.json) · [Environment/source hashes](ENVIRONMENT.json) · [English setup](README.md) · [中文使用](README.zh-CN.md).

Further prefill optimization measured median898.117 ms (ten updates, all below1 s), but one additional waveform date guard failed; candidate11/12, **not adopted**. The existing deployment is retained. [Experimental report](native-injection/PREFILL_RESULTS.md), [candidate summary](native-injection/PREFILL_SUMMARY.json), [latest environment](native-injection/PREFILL_ENVIRONMENT.json).

Two complete paced synthetic-speech flows (33/34) passed eight key waveform checks before confirmation, including speaking-time correction and same-reference rescheduling. Both had one model handshake and zero device fallback. A separate direct tool-fact probe (35, zero user PCM) passed Tuesday11AM/Wednesday9:30AM proposals and receipts plus cancellation during prefill. Total **12 key waveform checks across four date/time values**. Main entry8769 additionally passed native protocol/cache/PCM and independent waveform verification using labeled diagnostic text plus paced zero PCM.

Ten tool-update → first same-epoch voiced gateway PCM measurements were **1474.007–1734.432 ms, median1551.531 ms**, previously3338.1645 ms (**53.52% lower**). These exclude observer/query waits, browser/hardware and strict cold start. Whole flows55.214/55.034 s include independent ASR validation and are not single-turn latency.

Guided sampled-date mode remains experimental: it passed two full flows but failed additional actual-waveform checks, corroborated by Vosk and three repeated Qwen decodes of identical PCM. It is not the selected Demo mode. [Earlier native results](native-injection/RESULTS.md), [pre-cache summary](native-injection/PRE_CACHE_SUMMARY.json), [pre-cache environment](native-injection/PRE_CACHE_ENVIRONMENT.json) and [device baseline](native-injection/BASELINE_REPORT.md) preserve prior measurements and failures; changing protocols are not pooled into a success rate.

At the previous selected release: 83 scheduling tests, 618 full-suite tests (1 skip, 22 subtests; 3 existing concurrent DuplexLab NumPy warnings) and12 package/distribution checks passed. Nine runtime source hashes and active/reproduction backend hashes match. Gateway8768/backend8998 remain loopback-only behind local forward8769; old8999, candidate8767 and forward8766 stopped.

Human microphone/headset, acoustic echo, subjective quality, arbitrary expressions and stable p95 remain pending. These small development samples are not a frozen benchmark or production-readiness claim. The UI/tool receipt is authoritative. Only the isolated H100 GPU0 was used; no apex, training, subagents, commit or push.

Latest prefill work: 103 focused scheduling/distribution checks and633 full-suite tests passed (2 skipped,22 subtests;3 existing concurrent DuplexLab NumPy warnings), Ruff F clean. Both backend variants reproduced their hashes.16 related Markdown documents/277 local links checked, no missing targets. Candidate source frozen and remote disk hashes verified; the default processes retain the previous frozen runtime.
