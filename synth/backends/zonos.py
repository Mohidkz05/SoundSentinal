"""Zonos v0.1 transformer (Zyphra, Apache-2.0). Voice cloning from a speaker
embedding of the prompt."""

from ._util import load_mono, to_numpy


# The pip-installed package omits zonos/backbone, so the source at a pinned
# commit is put first on sys.path; pip still supplies the dependencies.
ZONOS_COMMIT = "bc40d98e1e1ab54fc65c483be127a90e3c7c0645"


def _zonos_source():
    import os
    import sys
    import tarfile
    import urllib.request
    root = os.path.join(os.environ.get("HF_HOME", "."), f"zonos-src-{ZONOS_COMMIT[:12]}")
    if not os.path.isdir(root):
        tgz = root + ".tar.gz"
        urllib.request.urlretrieve(
            f"https://github.com/Zyphra/Zonos/archive/{ZONOS_COMMIT}.tar.gz", tgz)
        with tarfile.open(tgz) as t:
            t.extractall(root)
    sys.path.insert(0, os.path.join(root, f"Zonos-{ZONOS_COMMIT}"))


def load(model_id, repo):
    from ._util import use_bundled_espeak
    use_bundled_espeak()
    _zonos_source()
    import torch
    import zonos.speaker_cloning as sc
    from zonos.model import Zonos
    # Zonos builds its speaker encoder's mel filterbank inside nested
    # torch.device() contexts, where torchaudio ends up mixing CPU and GPU
    # tensors (seen with torchaudio 2.8 and 2.11). Build just the filterbank
    # on the CPU, then move it to the GPU with the rest of the encoder.
    orig_init = sc.logFbankCal.__init__

    def init_on_cpu(self, *args, **kwargs):
        with torch.device("cpu"):
            orig_init(self, *args, **kwargs)
        self.to("cuda")

    sc.logFbankCal.__init__ = init_on_cpu
    return dict(model=Zonos.from_pretrained(repo, device="cuda"), speakers={})


def synthesize(m, job):
    import torch
    from zonos.conditioning import make_cond_dict
    model = m["model"]
    # One embedding per prompt file, computed once.
    if job["prompt_path"] not in m["speakers"]:
        wav, sr = load_mono(job["prompt_path"])
        m["speakers"][job["prompt_path"]] = model.make_speaker_embedding(
            torch.from_numpy(wav).unsqueeze(0).to("cuda"), sr)
    cond = make_cond_dict(text=job["text"], speaker=m["speakers"][job["prompt_path"]],
                          language="en-us")
    with torch.no_grad():
        codes = model.generate(model.prepare_conditioning(cond))
        wav = model.autoencoder.decode(codes).cpu()[0]
    return to_numpy(wav), model.autoencoder.sampling_rate, job["speaker"]
