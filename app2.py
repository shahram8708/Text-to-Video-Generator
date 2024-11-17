import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

device = "cuda" if torch.cuda.is_available() else "cpu" 
dtype = torch.float16 if device == "cuda" else torch.float32

step = 4 
repo = "ByteDance/AnimateDiff-Lightning"
ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
base = "emilianJR/epiCRealism" 

adapter = MotionAdapter().to(device, dtype)
adapter.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))

pipe = AnimateDiffPipeline.from_pretrained(
    base,
    motion_adapter=adapter,
    torch_dtype=dtype
).to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config,
    timestep_spacing="trailing",
    beta_schedule="linear"
)

output = pipe(
    prompt="A futuristic cityscape at night, glowing neon skyscrapers reflecting in a river, with flying cars zooming past and a serene moon in the sky, vibrant colors, cinematic lighting, and a sense of wonder and technological marvel.",
    guidance_scale=1.0,
    num_inference_steps=step
)

export_to_gif(output.frames[0], "animation.gif")

from google.colab import files
files.download("animation.gif")
