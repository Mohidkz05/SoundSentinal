"""VoxCPM 0.5B and 1.5 (OpenBMB, Apache-2.0). Voice cloning from the prompt and
its transcript. The optional denoiser (a ModelScope model) is not loaded."""

from ._util import to_numpy


def load(model_id, repo):
    from voxcpm import VoxCPM
    return VoxCPM.from_pretrained(repo, load_denoiser=False)


def synthesize(model, job):
    wav = model.generate(text=job["text"], prompt_wav_path=job["prompt_path"],
                         prompt_text=job["prompt_text"], cfg_value=2.0,
                         inference_timesteps=10, normalize=False, denoise=False,
                         retry_badcase=True)
    return to_numpy(wav), model.tts_model.sample_rate, job["speaker"]
