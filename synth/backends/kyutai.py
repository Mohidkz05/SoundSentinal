"""Kyutai TTS 1.6B en/fr (Kyutai, CC BY 4.0) via moshi. Preset voices: only the
embeddings in kyutai/tts-voices whose recordings allow commercial use — the
CC0 voice donations and VCTK (CC BY 4.0). Expresso (CC BY-NC) is excluded."""

ALLOWED = ("voice-donations/", "vctk/")


def load(model_id, repo):
    from huggingface_hub import list_repo_files
    from moshi.models.loaders import CheckpointInfo
    from moshi.models.tts import DEFAULT_DSM_TTS_VOICE_REPO, TTSModel
    tts = TTSModel.from_checkpoint_info(CheckpointInfo.from_hf_repo(repo), n_q=32, temp=0.6,
                                        device="cuda")
    voices = sorted(f[:-len(".1e68beda@240.safetensors")]
                    for f in list_repo_files(DEFAULT_DSM_TTS_VOICE_REPO)
                    if f.startswith(ALLOWED) and f.endswith(".1e68beda@240.safetensors")
                    and "_enhanced" not in f)
    if not voices:
        raise RuntimeError("no commercially licensed Kyutai voices found")
    return dict(tts=tts, voices=voices)


def synthesize(m, job):
    import numpy as np
    import torch
    tts = m["tts"]
    voice = m["voices"][job["index"] % len(m["voices"])]
    entries = tts.prepare_script([job["text"]], padding_between=1)
    attrs = tts.make_condition_attributes([tts.get_voice_path(voice)], cfg_coef=2.0)
    result = tts.generate([entries], [attrs])
    with tts.mimi.streaming(1), torch.no_grad():
        pcm = [tts.mimi.decode(f[:, 1:, :]).cpu().numpy()[0, 0]
               for f in result.frames[tts.delay_steps:]]
    return np.clip(np.concatenate(pcm), -1, 1), tts.mimi.sample_rate, voice
