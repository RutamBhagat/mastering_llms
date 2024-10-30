# %%
# Import necessary libraries
import gradio as gr
import openai
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    BitsAndBytesConfig,
)
from IPython.display import Markdown, display, update_display

# %%
# Constants
AUDIO_MODEL = "whisper-1"
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# %% [markdown]
# ### Mount Google Drive
# This section mounts Google Drive to access audio files directly.

# %%
from google.colab import drive

drive.mount("/content/drive")

# %%
# Load the audio file
audio_filename = "/content/drive/MyDrive/llms/denver_extract.mp3"

# %% [markdown]
# ### Define Gradio Interface
# The function below loads the audio file, sends it to OpenAI Whisper for transcription, and generates meeting minutes using the LLaMA model.


# %%
# Define Gradio interface function
def generate_minutes(audio):
    # OpenAI Whisper transcription
    with open(audio, "rb") as audio_file:
        transcription = openai.Audio.transcribe(
            model=AUDIO_MODEL, file=audio_file, response_format="text"
        )

    # Prepare for GPT-4 or similar model
    system_message = "Produce minutes of meetings in markdown with summary, discussion points, takeaways, and action items."
    user_prompt = f"Here is a transcript from a council meeting:\n{transcription}"

    prompts = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]

    # %% [markdown]
    # #### Stream back results in markdown

    stream = openai.ChatCompletion.create(
        model="gpt-4o", messages=prompts, temperature=0.7, stream=True
    )

    reply = ""
    display_handle = display(Markdown(""), display_id=True)
    for chunk in stream:
        reply += chunk.choices[0].delta.content or ""
        reply = reply.replace("```", "").replace("markdown", "")
        update_display(Markdown(reply), display_id=display_handle.display_id)

    return reply


# %%
# Setup Gradio UI
gr.Interface(
    fn=generate_minutes,
    inputs="audio",
    outputs="markdown",
    title="Meeting Minutes Generator",
    description="Upload an audio file to generate meeting minutes.",
).launch(share=True)

# %% [markdown]
# ### Optional: Saving as Markdown
# The output can be saved as a Markdown file or downloaded after processing.
