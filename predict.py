# Cog predictor for SkyTNT/midi-model (tv2o-medium) → Replicate.
#
# DROP THIS FILE INTO A CLONE/FORK OF github.com/SkyTNT/midi-model
# (repo root) so that `from midi_model import ...`, `import MIDI`, and
# `from midi_tokenizer import ...` resolve against the vendored modules.
#
# Weights are baked into the image at BUILD time (see cog.yaml build.run),
# so setup() loads them from the local HF cache — no network at cold start,
# and no boot-time download that could blow Replicate's health-check timeout.
#
# KEY PRODUCT NOTE: skytnt is NOT text-conditioned. It is conditioned on
# structured controls (instruments / bpm / time_signature / key_signature).

import time
import traceback
from typing import Optional

import numpy as np
import torch
from cog import BasePredictor, Input, Path
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safe_load_file

import MIDI
from midi_model import MIDIModel, MIDIModelConfig

HF_MODEL_ID = "skytnt/midi-model-tv2o-medium"
MODEL_CONFIG = "tv2o-medium"


class Predictor(BasePredictor):
    def setup(self):
        """Load tv2o-medium weights (baked into the image) into the GPU once."""
        t0 = time.time()
        print("[setup] start", flush=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[setup] device={self.device} cuda_available={torch.cuda.is_available()}", flush=True)

        config = MIDIModelConfig.from_name(MODEL_CONFIG)
        self.model = MIDIModel(config=config)
        print("[setup] model constructed; resolving weights from HF cache...", flush=True)

        # hf_hub_download hits the local cache first (weights baked at build time).
        ckpt_path = hf_hub_download(HF_MODEL_ID, "model.safetensors")
        print(f"[setup] weights at {ckpt_path}; loading state_dict...", flush=True)
        state_dict = safe_load_file(ckpt_path)
        result = self.model.load_state_dict(state_dict, strict=False)
        missing = getattr(result, "missing_keys", [])
        unexpected = getattr(result, "unexpected_keys", [])
        print(f"[setup] state_dict loaded. missing={len(missing)} unexpected={len(unexpected)}", flush=True)

        self.model.to(self.device).eval()
        self.tokenizer = self.model.tokenizer
        print(f"[setup] done in {time.time() - t0:.1f}s", flush=True)

    @torch.inference_mode()
    def predict(
        self,
        instruments: str = Input(
            description="Comma-separated General MIDI program numbers (0-127) for the "
            "non-drum tracks, e.g. '0,33,48' (Piano, Bass, Strings). Order = track order.",
            default="0",
        ),
        add_drums: bool = Input(
            description="Add a standard drum kit on channel 10.", default=True
        ),
        bpm: int = Input(
            description="Tempo in beats per minute. 0 = let the model decide.",
            default=120, ge=0, le=300,
        ),
        time_sig_numerator: int = Input(
            description="Time signature numerator (e.g. 4 for 4/4). 0 = unset.",
            default=4, ge=0, le=16,
        ),
        time_sig_denominator: int = Input(
            # NOTE: do NOT use choices=[...] here. Under this Cog/pydantic
            # version it makes the delivered value an unhashable list, which
            # blows up the dict lookup below. Plain int + in-code validation.
            description="Time signature denominator: 2, 4, 8 or 16. 0 = unset.",
            default=4, ge=0, le=16,
        ),
        key_sig_sharps_flats: int = Input(
            description="Key signature: sharps (+) / flats (-), -7..7.",
            default=0, ge=-7, le=7,
        ),
        key_sig_minor: bool = Input(
            description="Minor key (true) or major (false).", default=False
        ),
        max_len: int = Input(
            description="Max generation length in tokens. Higher = longer/denser piece.",
            default=512, ge=64, le=2048,
        ),
        temperature: float = Input(default=1.0, ge=0.1, le=2.0),
        top_p: float = Input(default=0.98, ge=0.1, le=1.0),
        top_k: int = Input(default=20, ge=1, le=128),
        seed: int = Input(description="Random seed. -1 = random.", default=-1),
    ) -> Path:
        t0 = time.time()
        try:
            tokenizer = self.tokenizer

            gen = None
            if seed is not None and seed >= 0:
                gen = torch.Generator(device=self.device).manual_seed(int(seed))

            # ---- Build the seed token sequence (mirrors app.py run(), tab==0) ----
            mid = [[tokenizer.bos_id] + [tokenizer.pad_id] * (tokenizer.max_token_seq - 1)]

            tsn = int(time_sig_numerator)
            tsd = int(time_sig_denominator)
            if getattr(tokenizer, "version", "v2") == "v2":
                dd_map = {2: 1, 4: 2, 8: 3, 16: 4}
                if tsn and tsd in dd_map:
                    dd = dd_map[tsd]
                    mid.append(
                        tokenizer.event2tokens(
                            ["time_signature", 0, 0, 0, tsn - 1, dd - 1]
                        )
                    )
                mid.append(
                    tokenizer.event2tokens(
                        ["key_signature", 0, 0, 0, key_sig_sharps_flats + 7, int(key_sig_minor)]
                    )
                )

            if bpm and bpm > 0:
                mid.append(tokenizer.event2tokens(["set_tempo", 0, 0, 0, bpm]))

            patches = {}
            ch = 0
            for tok in str(instruments).split(","):
                tok = tok.strip()
                if tok == "":
                    continue
                program = max(0, min(127, int(tok)))
                patches[ch] = program
                ch = (ch + 1) if ch != 8 else 10  # skip channel 9 (drums), per app.py
            if add_drums:
                patches[9] = 0  # standard kit

            for idx, (c, p) in enumerate(patches.items()):
                mid.append(tokenizer.event2tokens(["patch_change", 0, 0, idx + 1, c, p]))

            # event2tokens returns [] for out-of-range params; drop those so the
            # seed stays a rectangular (N, max_token_seq) array for np.asarray.
            seed_seq = [ev for ev in mid if ev]
            mid_arr = np.asarray([seed_seq], dtype=np.int64)  # batch_size = 1
            print(f"[predict] seed events={len(seed_seq)} max_len={max_len}", flush=True)

            # ---- Generate ----
            # MIDIModel.generate (a transformers PreTrainedModel subclass) is NOT
            # the streaming app.py generate(): it runs to completion and RETURNS
            # the full token array of shape (batch, total_len, max_token_seq).
            # It also does not accept the app.py disable_* flags.
            output = self.model.generate(
                mid_arr, batch_size=1, max_len=max_len, temp=temperature,
                top_p=top_p, top_k=top_k, generator=gen,
            )
            out_arr = np.asarray(output)
            full_seq = out_arr[0].tolist()  # seed + generated token-lists
            print(f"[predict] generated total events={len(full_seq)}", flush=True)

            # ---- Detokenize → standard MIDI bytes ----
            score = tokenizer.detokenize(full_seq)
            midi_bytes = MIDI.score2midi(score)

            out_path = "/tmp/output.mid"
            with open(out_path, "wb") as f:
                f.write(midi_bytes)
            print(f"[predict] wrote {len(midi_bytes)} bytes in {time.time() - t0:.1f}s", flush=True)
            return Path(out_path)
        except Exception:
            tb = traceback.format_exc()
            print("[predict] FAILED:\n" + tb, flush=True)
            # Surface the full traceback in Replicate's `error` field so the exact
            # failing line is visible even when container logs are not captured.
            raise RuntimeError("predict failed:\n" + tb)
