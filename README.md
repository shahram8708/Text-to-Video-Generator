# Video Generation and Animation with Diffusers

This repository demonstrates how to generate videos and animations using pretrained models from the Hugging Face `diffusers` library. The repository contains two main sections:

1. **Text-to-Video Generation**: Generate a video from a text prompt.
2. **AnimateDiff Animation**: Create an animation based on a text prompt using a pretrained animation model.

## Prerequisites

To run the code, install the required dependencies:

```bash
pip install torch torchvision diffusers huggingface_hub safetensors PIL google-colab
```

## 1. Text-to-Video Generation

This section allows you to generate a video by providing a text description.

### How It Works:

- **Model**: Uses the `damo-vilab/text-to-video-ms-1.7b` model from Hugging Face to generate videos.
- **Functionality**: The code takes a text prompt, processes it through the model, and generates a video with frames that are exported to a video file.

### How to Use:

1. Clone the repository or run the script on Google Colab.
2. Ensure the required dependencies are installed.
3. Run the script and provide your desired text prompt when asked.
4. The video will be generated and saved to the disk.

## 2. AnimateDiff Animation

This section generates an animation using the AnimateDiff model based on a text prompt.

### How It Works:

- **Model**: Uses the `AnimateDiff` model, which integrates motion adaptation for dynamic animations.
- **Functionality**: The code takes a text prompt, processes it through the animation model, and exports the result as a GIF file.

### How to Use:

1. Clone the repository or run the script on Google Colab.
2. Ensure the required dependencies are installed.
3. Run the script, and the generated animation will be automatically downloaded as a GIF.
