import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from PIL import Image

def load_video_model(model_id):
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
    pipe.enable_vae_slicing()
    return pipe

def generate_video(pipe, prompt, num_frames=24):
    video_frames = pipe(prompt, num_frames=num_frames).frames[0]
    video_path = export_to_video(video_frames)
    return video_path

def main():
    model_id = "damo-vilab/text-to-video-ms-1.7b" 
    pipe = load_video_model(model_id)
    
    prompt = input("Enter text to generate video: ")
    
    video_path = generate_video(pipe, prompt)
    
    print(f"Video generated and saved to: {video_path}")

main()
