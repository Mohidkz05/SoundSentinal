"""SpeechT5 TTS with its HiFi-GAN vocoder (Microsoft, MIT). Preset voices: the
x-vectors of the seven CMU ARCTIC speakers (Matthijs/cmu-arctic-xvectors, MIT)."""

import numpy as np


def load(model_id, repo):
    import zipfile
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor
    z = zipfile.ZipFile(hf_hub_download("Matthijs/cmu-arctic-xvectors", "spkrec-xvect.zip",
                                        repo_type="dataset"))
    names = sorted(n for n in z.namelist() if n.endswith(".npy"))
    # Seven ARCTIC speakers, ~1,130 x-vectors each (one per utterance). A clip
    # takes speaker index % 7 and that speaker's (index // 7)-th x-vector, so
    # the voices vary within a speaker as well as across the seven.
    import io
    import re
    xvecs = {}
    for n in names:
        spk = re.search(r"cmu_us_([a-z]+)_arctic", n).group(1)
        xvecs.setdefault(spk, []).append(
            torch.from_numpy(np.load(io.BytesIO(z.read(n)))).reshape(1, -1))
    return dict(processor=SpeechT5Processor.from_pretrained(repo),
                model=SpeechT5ForTextToSpeech.from_pretrained(repo).cuda().eval(),
                vocoder=SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").cuda(),
                xvecs=xvecs)


def synthesize(m, job):
    import torch
    speakers = sorted(m["xvecs"])
    spk = speakers[job["index"] % len(speakers)]
    vecs = m["xvecs"][spk]
    k = (job["index"] // len(speakers)) % len(vecs)
    inputs = m["processor"](text=job["text"], return_tensors="pt")
    with torch.no_grad():
        speech = m["model"].generate_speech(inputs["input_ids"].cuda(), vecs[k].cuda(),
                                            vocoder=m["vocoder"])
    return speech.cpu().numpy(), 16000, f"arctic-{spk}-{k}"
