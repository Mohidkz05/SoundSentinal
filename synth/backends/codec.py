"""Genuine speech re-encoded through an open neural codec (RESULTS.md Finding
19). Not a TTS model: each job's own LibriSpeech utterance (text_path) is
resampled to the codec's rate, encoded, decoded, and written as a fake, so
the only thing that differs from the real recording is the codec's decoder.

    snac-24khz        SNAC 24 kHz (Siuzdak, MIT), pip package `snac`
    wavtokenizer-75   WavTokenizer large, 75 tokens/s (Ji et al., MIT). Not on
                      PyPI: the repo is cloned at a pinned commit beside this
                      environment; its weights are the MIT checkpoint on
                      Hugging Face (its encoder is EnCodec's *code*, not
                      EnCodec's weights).
"""

import os
import subprocess
import sys
from pathlib import Path

from ._util import load_mono

SR = 24000
WAVTOKENIZER_REPO = "https://github.com/jishengpeng/WavTokenizer"
WAVTOKENIZER_COMMIT = "5cf440d91ac420ca338f117b7003a77450d64730"
WAVTOKENIZER_CONFIG = "configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
WAVTOKENIZER_CKPT = "wavtokenizer_large_speech_320_v2.ckpt"


def _wavtokenizer_source():
    """The pinned checkout, cloned once next to the running environment."""
    dest = Path(sys.prefix) / "src" / f"WavTokenizer-{WAVTOKENIZER_COMMIT[:12]}"
    if not (dest / "decoder" / "pretrained.py").is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", WAVTOKENIZER_REPO, str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "-q", WAVTOKENIZER_COMMIT],
                       check=True)
    return dest


def load(model_id, repo):
    import torch
    if model_id == "snac-24khz":
        from snac import SNAC
        return dict(kind="snac", codec=SNAC.from_pretrained(repo).eval().cuda())
    if model_id == "wavtokenizer-75":
        from huggingface_hub import hf_hub_download
        src = _wavtokenizer_source()
        sys.path.insert(0, str(src))
        from decoder.pretrained import WavTokenizer
        ckpt = hf_hub_download(repo, WAVTOKENIZER_CKPT)
        # A Lightning checkpoint holds non-tensor objects; torch >= 2.6 refuses
        # those by default, and the repo's loader does not pass the flag.
        load_ = torch.load
        torch.load = lambda *a, **k: load_(*a, **{**k, "weights_only": False})
        try:
            codec = WavTokenizer.from_pretrained0802(str(src / WAVTOKENIZER_CONFIG), ckpt)
        finally:
            torch.load = load_
        return dict(kind="wavtokenizer", codec=codec.eval().cuda())
    raise ValueError(f"unknown codec {model_id}")


def synthesize(m, job):
    import torch
    x, _ = load_mono(job["text_path"], SR)
    wav = torch.from_numpy(x).cuda()
    with torch.inference_mode():
        if m["kind"] == "snac":
            # SNAC pads the input to its hop internally; trim back to length.
            codes = m["codec"].encode(wav.view(1, 1, -1))
            out = m["codec"].decode(codes)[0, 0, :len(x)]
        else:
            bw = torch.tensor([0], device="cuda")
            features, _ = m["codec"].encode_infer(wav.view(1, -1), bandwidth_id=bw)
            out = m["codec"].decode(features, bandwidth_id=bw)[0, :len(x)]
    return out.float().cpu().numpy(), SR, os.path.basename(job["text_path"])
