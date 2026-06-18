# Cog predictor for SkyTNT/midi-model (tv2o-medium) → Replicate.
#
# DROP THIS FILE INTO A CLONE/FORK OF github.com/SkyTNT/midi-model
# (repo root) so that `from midi_model import ...`, `import MIDI`, and
# `from midi_tokenizer import ...` resolve against the vendored modules.
#
# ⚠️ FIRST-BUILD STATUS: this is written against the repo's app.py `run()`
# logic (model load → seed tokens → drain generate() → detokenize →
# score2midi) but has NOT yet been validated by a real `cog push`. Expect
# 1–4 rounds of dependency / API tweaks on the first build (see README §6).
#
# KEY PRODUCT NOTE: skytnt is NOT text-conditioned. It is conditioned on
# structured controls (instruments / bpm / time_signature / key_signature).
# That is exactly the "Advanced" engine's contract in AI MIDI Studio.

import os
from typing import List, Optional

import numpy as np
import torch
from cog import BasePredictor, Input, Path

import MIDI
from midi_model import MIDIModel, MIDIModelConfig
from midi_tokenizer import MIDITokenizer  # noqa: F401  (version checked via model.tokenizer)

HF_MODEL_ID = "skytnt/midi-model-tv2o-medium"
MODEL_CONFIG = "tv2o-medium"

# General MIDI program names → program number. skytnt's app.py builds these
# maps from MIDI.py's Number2patch; we mirror a compact, commonly-used subset
# and resolve the rest by GM program number at request time.
# Channel assignment follows app.py: channels 0..7 then 10.. (9 reserved drums).


class Predictor(BasePredictor):
    def setup(self):
        """Load tv2o-medium weights into GPU once per cold start."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config = MIDIModelConfig.from_name(MODEL_CONFIG)
        # MIDIModel subclasses transformers.PreTrainedModel → from_pretrained
        # pulls config.json + model.safetensors from the public HF repo.
        self.model = MIDIModel.from_pretrained(HF_MODEL_ID, config=config)
        self.model.to(self.device).eval()
        self.tokenizer = self.model.tokenizer

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
            default=120,
            ge=0,
            le=300,
        ),
        time_sig_numerator: int = Input(
            description="Time signature numerator (e.g. 4 for 4/4). 0 = unset.",
            default=4,
            ge=0,
            le=16,
        ),
        time_sig_denominator: int = Input(
            description="Time signature denominator: 2, 4, 8 or 16. 0 = unset.",
            default=4,
            choices=[0, 2, 4, 8, 16],
        ),
        key_sig_sharps_flats: int = Input(
            description="Key signature: sharps (+) / flats (-), -7..7. Ignored if unset.",
            default=0,
            ge=-7,
            le=7,
        ),
        key_sig_minor: bool = Input(
            description="Minor key (true) or major (false).", default=False
        ),
        max_len: int = Input(
            description="Max generation length in tokens. Higher = longer/denser piece.",
            default=512,
            ge=64,
            le=2048,
        ),
        temperature: float = Input(default=1.0, ge=0.1, le=2.0),
        top_p: float = Input(default=0.98, ge=0.1, le=1.0),
        top_k: int = Input(default=20, ge=1, le=128),
        seed: int = Input(
            description="Random seed. -1 = random.", default=-1
        ),
    ) -> Path:
        tokenizer = self.tokenizer

        gen = None
        if seed is not None and seed >= 0:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))

        # ---- Build the seed token sequence (mirrors app.py run(), tab==0) ----
        mid = [[tokenizer.bos_id] + [tokenizer.pad_id] * (tokenizer.max_token_seq - 1)]

        if getattr(tokenizer, "version", "v2") == "v2":
            if time_sig_numerator and time_sig_denominator:
                # denominator encoded as power of two index (dd): 2->1,4->2,8->3,16->4
                dd = {2: 1, 4: 2, 8: 3, 16: 4}[time_sig_denominator]
                mid.append(
                    tokenizer.event2tokens(
                        ["time_signature", 0, 0, 0, time_sig_numerator - 1, dd - 1]
                    )
                )
            # key signature is always meaningful (0 sharps/flats = C major / A minor)
            mid.append(
                tokenizer.event2tokens(
                    ["key_signature", 0, 0, 0, key_sig_sharps_flats + 7, int(key_sig_minor)]
                )
            )

        if bpm and bpm > 0:
            mid.append(tokenizer.event2tokens(["set_tempo", 0, 0, 0, bpm]))

        # patch_change events: assign instruments to channels 0..7,10,11.. (9=drums)
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

        seed_seq = mid  # list of token-sequences
        mid_arr = np.asarray([seed_seq], dtype=np.int64)  # batch_size = 1

        # ---- Drain the autoregressive generator (mirrors app.py loop) ----
        mid_seq = [list(seed_seq)]  # accumulate per batch element
        midi_generator = self.model.generate(
            mid_arr,
            batch_size=1,
            max_len=max_len,
            temp=temperature,
            top_p=top_p,
            top_k=top_k,
            disable_patch_change=False,
            disable_control_change=False,
            disable_channels=None,
            generator=gen,
        )
        for token_seqs in midi_generator:
            token_seqs = token_seqs.tolist()
            mid_seq[0].append(token_seqs[0])

        # ---- Detokenize → standard MIDI bytes ----
        score = tokenizer.detokenize(mid_seq[0])
        midi_bytes = MIDI.score2midi(score)

        out_path = "/tmp/output.mid"
        with open(out_path, "wb") as f:
            f.write(midi_bytes)
        return Path(out_path)
