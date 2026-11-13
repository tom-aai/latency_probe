# Turn-detection latency probe

This probe compares how fast two AssemblyAI streaming models detect end of turn.
It runs the models side by side on the same live microphone audio. There is no
voice agent, no LLM, and no TTS. You speak, and the probe measures the STT turn
stage only.

## What it measures

For each turn, the probe reports one number:

```
detect = (time the end-of-turn signal reached the probe)
         - (wall-clock time your last word ended)
```

The second term comes from the plugin's `END_OF_SPEECH` event. The vendored
plugin pins its `speech_end_time` to the last word's audio timestamp, not to the
moment the final transcript arrived. So this window includes three things: the
provider's silence wait, its end-of-turn decision, and the network and
transcription delay.

This is the same quantity the agent prints as `end_of_turn_delay`. The agent
then adds LiveKit endpointing on top. The probe leaves that out, so the number
is the model's own latency.

Both models get the **same** turn-detection config. So the difference you see
is the model, not the settings.

## Why the measurement is done this way

A naive probe starts the clock when the final transcript arrives. That drops the
transcription delay and makes every model look near-instant. The last-word anchor
is the fix. It is the same anchor the production agent uses. See the
`END_OF_SPEECH` block in `assemblyai_stt/stt.py`.

## What is in this repo

The probe is self-contained, so you can clone it and run it on its own:

- `turn_latency_probe.py` — the probe.
- `assemblyai_stt/` — a vendored copy of the project's AssemblyAI streaming STT
  plugin. It adds the `universal-3-7-preview` model and its Pro options, which
  the published `livekit-plugins-assemblyai` does not expose. The probe imports
  this local copy, not the published plugin.
- `pyproject.toml` — the dependencies, managed by [uv](https://docs.astral.sh/uv/).

## Run it

You need [uv](https://docs.astral.sh/uv/) and an AssemblyAI API key.

1. Clone the repo and enter it:

   ```console
   git clone git@github.com:tom-aai/latency_probe.git
   cd latency_probe
   ```

2. Add your key. Copy the example file and fill in `ASSEMBLYAI_API_KEY`:

   ```console
   cp .env.example .env.local
   ```

   Get a key at <https://www.assemblyai.com/app/account>. The `.env.local` file
   is git-ignored, so your key stays out of the repo.

3. Run the probe. uv installs the dependencies on the first run:

   ```console
   uv run turn_latency_probe.py
   ```

Speak a few complete sentences. Pause between them. Each pause that a model
treats as end of turn prints one line per model. Press Ctrl+C to stop and print
the summary.

Example output:

```
10:14:02  universal-3-6-pro      turn  1  detect    612.4 ms  "my favorite restaurant is hiroshi's"
10:14:02  universal-3-7-preview  turn  1  detect    318.9 ms  "my favorite restaurant is hiroshi's"
```

The summary at the end gives per-model mean, median, p90, min, and max, and a
per-turn table that pairs the two models and shows the delta.

## Options

| Flag | Default | What it does |
|------|---------|--------------|
| `--models` | `universal-3-6-pro universal-3-7-preview` | Models to run in parallel, baseline first. |
| `--min-turn-silence` | `100` | Silence (ms) before a confident end of turn is allowed. Shared by all models. |
| `--max-turn-silence` | `1000` | Silence (ms) that always ends a turn. Shared by all models. |
| `--mode` | (none) | Preset: `min_latency`, `balanced`, or `max_accuracy`. When set, it governs the silence tuning unless you also pass the silence flags. |
| `--sample-rate` | `16000` | Capture sample rate in Hz. |
| `--device` | default mic | Input device index or name for sounddevice. |
| `--list-devices` | | List audio input devices and exit. |

To hold the comparison controlled, keep the silence flags the same for both
models (the default does this). To reproduce the agent's production 3.7 setting,
the defaults already match it: `min_turn_silence=100`, `max_turn_silence=1000`.

## Notes on reading the numbers

- `detect` includes the silence wait. With `max_turn_silence=1000`, a turn the
  model is unsure about can wait up to a second before it ends. A confident turn
  can end at `min_turn_silence` (100 ms) plus processing. The gap between the two
  models comes mostly from how fast each one gets confident.
- Say several turns. One turn is noise. Ten turns show the real difference.
- The two models open separate AssemblyAI sessions, so they get the same audio
  but decide on their own.
