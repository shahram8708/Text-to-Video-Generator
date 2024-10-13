import streamlit as st
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from PIL import Image
import tempfile
import os

st.set_page_config(
    page_title="Text-to-Video Generator",
    page_icon="🎥",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("🎥 Text-to-Video Generator")
st.write("Generate stunning videos from simple text prompts using advanced AI models.")

if not torch.cuda.is_available():
    st.error("GPU is not available. Please ensure GPU is enabled.")
else:
    st.success(f"Using GPU: {torch.cuda.get_device_name(0)}")

@st.cache_resource  
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

model_id = "damo-vilab/text-to-video-ms-1.7b"
pipe = load_video_model(model_id)

prompt = st.text_input("Enter your text prompt to generate a video:")

if st.button("Generate Video"):
    if prompt:
        with st.spinner("Generating video... Please wait"):
            try:
                video_path = generate_video(pipe, prompt)
                st.success("Video generated successfully!")

                with open(video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)

                st.download_button(
                    label="Download Video",
                    data=video_bytes,
                    file_name="generated_video.mp4",
                    mime="video/mp4"
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a text prompt before generating a video.")
