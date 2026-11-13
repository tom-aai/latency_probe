"""Turn-detection latency probe for AssemblyAI streaming STT.

Run two (or more) AssemblyAI streaming models side by side on the same live
microphone audio and measure how fast each one signals end of turn after you
stop speaking. No voice agent, no LLM, no TTS: just the STT stage.

What it measures
----------------
For each turn, the number reported is:

    detect = (time the end-of-turn signal reached this process)
             - (wall-clock time your last word ended)

The second term comes from the plugin's END_OF_SPEECH event, whose
``speech_end_time`` the vendored plugin pins to the last word's audio
timestamp (not to when the final arrived). So this window includes the
provider's silence wait, its end-of-turn decision, and the network and
transcription delay. It is the same quantity the agent reports as
``end_of_turn_delay`` (the agent adds LiveKit endpointing on top, which this
probe leaves out so the number is the model's own latency).

Both models get the SAME turn-detection config, so the difference you see is
the model, not the settings. Override the config with the flags if you want.

Usage
-----
    uv run latency_probe/turn_latency_probe.py
    uv run latency_probe/turn_latency_probe.py --list-devices
    uv run latency_probe/turn_latency_probe.py --max-turn-silence 1000
    uv run latency_probe/turn_latency_probe.py --models universal-3-6-pro universal-3-7-preview

Speak a few complete sentences, pausing between them. Each pause that the model
treats as end of turn prints one line per model. Press Ctrl+C to stop and print
the summary.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import sounddevice as sd
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import stt as stt_base

import assemblyai_stt

# The project's agent reads its key from .env.local in the project root. This
# script lives one level down, so load the same file by its absolute path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env.local")

# AssemblyAI streaming wants 16 kHz mono 16-bit PCM. The plugin rebuffers into
# its own 50 ms frames, so the capture block size below only affects how often
# we hand it audio, not the result.
SAMPLE_RATE = 16000
CHANNELS = 1
CAPTURE_BLOCK = 1600  # samples per callback = 100 ms at 16 kHz


# ---------------------------------------------------------------------------
# Per-model collection
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """One finalized turn from one model."""

    index: int
    text: str
    detect_ms: float | None  # end-of-turn latency, None if no word timestamps
    confidence: float | None  # end_of_turn_confidence on the final


@dataclass
class ModelProbe:
    """Holds one model's stream and the turns it reports."""

    name: str
    stream: stt_base.SpeechStream
    turns: list[Turn] = field(default_factory=list)


def _fmt_ms(value: float | None) -> str:
    return f"{value:,.1f} ms" if value is not None else "n/a"


def _now_str() -> str:
    return time.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Building the models
# ---------------------------------------------------------------------------


def build_stt(
    model: str,
    *,
    http_session: aiohttp.ClientSession,
    min_turn_silence: int | None,
    max_turn_silence: int | None,
    mode: str | None,
    sample_rate: int,
) -> assemblyai_stt.STT:
    """Create one STT with the shared turn-detection config.

    When a ``mode`` preset is given it governs the silence tuning, so the
    explicit min/max are only passed when the caller set them. Otherwise both
    min and max turn silence are passed so every model waits the same.
    """
    kwargs: dict = {
        "api_key": None,  # falls back to ASSEMBLYAI_API_KEY in the environment
        "model": model,
        "sample_rate": sample_rate,
        "http_session": http_session,
    }
    # Drop api_key=None so the plugin reads the environment variable instead.
    del kwargs["api_key"]

    if mode is not None:
        kwargs["mode"] = mode
        if min_turn_silence is not None:
            kwargs["min_turn_silence"] = min_turn_silence
        if max_turn_silence is not None:
            kwargs["max_turn_silence"] = max_turn_silence
    else:
        if min_turn_silence is not None:
            kwargs["min_turn_silence"] = min_turn_silence
        if max_turn_silence is not None:
            kwargs["max_turn_silence"] = max_turn_silence

    return assemblyai_stt.STT(**kwargs)


# ---------------------------------------------------------------------------
# Consuming one model's event stream
# ---------------------------------------------------------------------------


async def consume(probe: ModelProbe) -> None:
    """Read one model's events and record each finalized turn.

    The plugin emits FINAL_TRANSCRIPT and then END_OF_SPEECH together for each
    end of turn. The final carries the text and the end-of-turn confidence; the
    END_OF_SPEECH carries ``speech_end_time``, the wall-clock moment the last
    word ended. The latency is "now minus that moment".
    """
    last_final_text = ""
    last_confidence: float | None = None

    async for ev in probe.stream:
        if ev.type == stt_base.SpeechEventType.FINAL_TRANSCRIPT:
            alt = ev.alternatives[0]
            last_final_text = alt.text
            last_confidence = (alt.metadata or {}).get("end_of_turn_confidence")

        elif ev.type == stt_base.SpeechEventType.END_OF_SPEECH:
            now = time.time()
            speech_end = ev.speech_end_time
            detect_ms = (now - speech_end) * 1000 if speech_end is not None else None

            turn = Turn(
                index=len(probe.turns) + 1,
                text=last_final_text,
                detect_ms=detect_ms,
                confidence=last_confidence,
            )
            probe.turns.append(turn)

            print(
                f"{_now_str()}  {probe.name:<22} "
                f"turn {turn.index:>2}  detect {_fmt_ms(detect_ms):>10}  "
                f'"{turn.text}"',
                flush=True,
            )
            last_final_text = ""
            last_confidence = None


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. `values` must be non-empty."""
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(pct / 100 * len(ordered) + 0.5) - 1))
    return ordered[k]


def print_summary(probes: list[ModelProbe]) -> None:
    print("\n" + "=" * 74)
    print("Turn-detection latency summary")
    print("=" * 74)

    # Per-model stats over the turns that produced a latency.
    header = (
        f"{'model':<22}{'turns':>6}{'mean':>11}{'median':>11}"
        f"{'p90':>11}{'min':>8}{'max':>9}"
    )
    print(header)
    print("-" * 74)
    for probe in probes:
        lat = [t.detect_ms for t in probe.turns if t.detect_ms is not None]
        if not lat:
            print(f"{probe.name:<22}{len(probe.turns):>6}   no timed turns")
            continue
        print(
            f"{probe.name:<22}{len(lat):>6}"
            f"{statistics.mean(lat):>10,.1f}m"
            f"{statistics.median(lat):>10,.1f}m"
            f"{_percentile(lat, 90):>10,.1f}m"
            f"{min(lat):>7,.0f}m"
            f"{max(lat):>8,.0f}m"
        )

    # Side-by-side per turn, paired by turn order. Only when exactly two models
    # ran, which is the common case (baseline vs. new).
    if len(probes) == 2:
        a, b = probes
        paired = min(len(a.turns), len(b.turns))
        if paired:
            print("\n" + "-" * 74)
            print(f"Per turn: {a.name}  vs  {b.name}  (delta = second minus first)")
            print("-" * 74)
            print(f"{'turn':>4}{a.name[:16]:>18}{b.name[:16]:>18}{'delta':>12}   text")
            deltas: list[float] = []
            for i in range(paired):
                ta, tb = a.turns[i], b.turns[i]
                da, db = ta.detect_ms, tb.detect_ms
                delta = (db - da) if (da is not None and db is not None) else None
                if delta is not None:
                    deltas.append(delta)
                text = ta.text or tb.text
                text = text if len(text) <= 28 else text[:27] + "…"
                print(
                    f"{i + 1:>4}{_fmt_ms(da):>18}{_fmt_ms(db):>18}"
                    f"{_fmt_ms(delta):>12}   {text}"
                )
            if deltas:
                mean_delta = statistics.mean(deltas)
                faster = b.name if mean_delta < 0 else a.name
                print("-" * 74)
                print(
                    f"Mean delta {mean_delta:+,.1f} ms over {len(deltas)} paired "
                    f"turns. On average {faster} detects end of turn sooner."
                )
    print("=" * 74, flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> None:
    models: list[str] = args.models
    print("Turn-detection latency probe")
    print(f"  models: {', '.join(models)}")
    if args.mode:
        print(f"  mode: {args.mode} (shared by all models)")
    cfg = []
    if args.min_turn_silence is not None:
        cfg.append(f"min_turn_silence={args.min_turn_silence}ms")
    if args.max_turn_silence is not None:
        cfg.append(f"max_turn_silence={args.max_turn_silence}ms")
    if cfg:
        print(f"  turn silence: {', '.join(cfg)} (shared by all models)")
    print(f"  sample rate: {args.sample_rate} Hz")
    print("\nSpeak now. Pause between sentences. Press Ctrl+C to stop.\n", flush=True)

    http_session = aiohttp.ClientSession()
    probes: list[ModelProbe] = []
    for model in models:
        stt_obj = build_stt(
            model,
            http_session=http_session,
            min_turn_silence=args.min_turn_silence,
            max_turn_silence=args.max_turn_silence,
            mode=args.mode,
            sample_rate=args.sample_rate,
        )
        probes.append(ModelProbe(name=model, stream=stt_obj.stream()))

    loop = asyncio.get_running_loop()
    audio_q: asyncio.Queue[bytes | None] = asyncio.Queue()
    stop = asyncio.Event()

    def on_audio(indata, _frames, _time_info, status) -> None:
        # Runs on sounddevice's thread. Hand the raw int16 PCM bytes to the
        # event loop; never touch asyncio objects directly from here.
        if status:
            print(f"[audio] {status}", flush=True)
        loop.call_soon_threadsafe(audio_q.put_nowait, bytes(indata))

    def request_stop() -> None:
        if not stop.is_set():
            stop.set()

    loop.add_signal_handler(signal.SIGINT, request_stop)
    with contextlib.suppress(NotImplementedError):
        loop.add_signal_handler(signal.SIGTERM, request_stop)

    consumers = [asyncio.create_task(consume(p)) for p in probes]

    audio_stream = sd.RawInputStream(
        samplerate=args.sample_rate,
        channels=CHANNELS,
        dtype="int16",
        blocksize=CAPTURE_BLOCK,
        device=args.device,
        callback=on_audio,
    )

    async def pump() -> None:
        """Wrap captured PCM in AudioFrames and feed every model the same audio."""
        while True:
            data = await audio_q.get()
            if data is None:
                return
            samples = len(data) // 2  # int16 mono
            frame = rtc.AudioFrame(
                data=data,
                sample_rate=args.sample_rate,
                num_channels=CHANNELS,
                samples_per_channel=samples,
            )
            for probe in probes:
                probe.stream.push_frame(frame)

    pump_task = asyncio.create_task(pump())

    try:
        with audio_stream:
            await stop.wait()
    finally:
        print("\nStopping. Flushing final turns...", flush=True)
        # Stop the pump, then close each model's input so any held turn finalizes.
        audio_q.put_nowait(None)
        with contextlib.suppress(asyncio.CancelledError):
            await pump_task
        for probe in probes:
            probe.stream.end_input()
        # Give the models a moment to emit any pending end-of-turn, then close.
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.gather(*consumers), timeout=3.0)
        for task in consumers:
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*consumers, return_exceptions=True)
        for probe in probes:
            await probe.stream.aclose()
        await http_session.close()

    print_summary(probes)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        nargs="+",
        default=["universal-3-6-pro", "universal-3-7-preview"],
        help="Models to run in parallel (baseline first). Default: 3.6-pro 3.7-preview.",
    )
    p.add_argument(
        "--min-turn-silence",
        type=int,
        default=100,
        help="Silence (ms) before a confident end of turn is allowed. Shared by all "
        "models. Default 100.",
    )
    p.add_argument(
        "--max-turn-silence",
        type=int,
        default=1000,
        help="Silence (ms) that always ends a turn. Shared by all models. Default 1000.",
    )
    p.add_argument(
        "--mode",
        choices=["min_latency", "balanced", "max_accuracy"],
        default=None,
        help="Optional preset. When set it governs silence tuning unless you also "
        "pass the silence flags.",
    )
    p.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    p.add_argument(
        "--device",
        default=None,
        help="Input device index or name for sounddevice. Omit for the default mic.",
    )
    p.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio input devices and exit.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return
    # Parse an integer device index when the user passed a number.
    if isinstance(args.device, str) and args.device.isdigit():
        args.device = int(args.device)
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
