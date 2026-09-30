"""SpeechT5 TTS with its HiFi-GAN vocoder (Microsoft, MIT). Preset voices: the
x-vectors of CMU ARCTIC speakers (Matthijs/cmu-arctic-xvectors, MIT), one per
clip in turn."""

import numpy as np


def load(model_id, repo):
    import zipfile
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor
    z = zipfile.ZipFile(hf_hub_download("Matthijs/cmu-arctic-xvectors", "spkrec-xvect.zip",
                                        repo_type="dataset"))
    names = sorted(n for n in z.namelist() if n.endswith(".npy"))
    # One x-vector per ARCTIC speaker (the file name starts with it), so the
    # voices differ rather than being 7,900 takes of eight people.
    by_speaker = {}
    for n in names:
        by_speaker.setdefault(n.split("/")[-1].split("_")[0], n)
    import io
    xvecs = {spk: torch.from_numpy(np.load(io.BytesIO(z.read(n)))).reshape(1, -1)
             for spk, n in sorted(by_speaker.items())}
    return dict(processor=SpeechT5Processor.from_pretrained(repo),
                model=SpeechT5ForTextToSpeech.from_pretrained(repo).cuda().eval(),
                vocoder=SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").cuda(),
                xvecs=xvecs)


def synthesize(m, job):
    import torch
    names = list(m["xvecs"])
    voice = names[job["index"] % len(names)]
    inputs = m["processor"](text=job["text"], return_tensors="pt")
    with torch.no_grad():
        speech = m["model"].generate_speech(inputs["input_ids"].cuda(),
                                            m["xvecs"][voice].cuda(), vocoder=m["vocoder"])
    return speech.cpu().numpy(), 16000, f"arctic-{voice}"
