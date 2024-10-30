# -*- coding: utf-8 -*-
"""Meeting Minutes Generation with Gradio Integration.

# This script generates meeting minutes from an audio file using a quantized language model.
"""

# %% [markdown]
# # Install dependencies
# Uncomment and run the following in Colab or if dependencies are not installed:
# ```shell
# !pip install -q gradio requests torch bitsandbytes transformers sentencepiece accelerate openai
# ```

# %%
# Imports
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    BitsAndBytesConfig,
    pipeline,
)
import gradio as gr
from IPython.display import Markdown, display, update_display

# %%
# Constants
AUDIO_MODEL = "whisper-1"  # Whisper model for transcription
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Quantized model for text generation

# %% [markdown]
# ### Mount Google Drive if needed
# Uncomment below if using Colab with audio files saved in Google Drive.

# %%
# from google.colab import drive
# drive.mount("/content/drive")

audio_filename = "/content/drive/MyDrive/llms/denver_extract.mp3"  # Update as needed

# %%
# Configure quantization with BitsAndBytesConfig
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

# Load tokenizer and quantized model
tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    LLAMA, device_map="auto", quantization_config=quant_config
)
model.to("cuda")  # Ensure model runs on GPU if available


# %%
# Define Gradio interface function
def generate_minutes(audio):
    # Transcribe audio using Whisper or local pipeline if Whisper unavailable
    audio_file = open(audio, "rb")
    transcription = openai.Audio.transcribe(
        model=AUDIO_MODEL, file=audio_file, response_format="text"
    )

    # Prepare prompt
    system_message = (
        "Generate minutes with summary, discussion points, and actions in markdown."
    )
    user_prompt = f"Transcript of a meeting:\n{transcription}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]

    # Tokenize and generate response
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
    streamer = TextStreamer(tokenizer)
    outputs = model.generate(inputs, max_new_tokens=2000, streamer=streamer)

    # Decode and return response
    response = tokenizer.decode(outputs[0])
    return response.replace("```", "").replace("markdown", "")


# %%
# Setup Gradio UI
gr.Interface(
    fn=generate_minutes,
    inputs="audio",
    outputs="markdown",
    title="Meeting Minutes Generator",
    description="Upload an audio file to generate meeting minutes with a quantized model.",
).launch(share=True)

# %% [markdown]
# ### Optional: Saving as Markdown
# This output can be saved as Markdown or text if needed.
