# %% [markdown]
# # Gradio Day!
#
# Today we will build User Interfaces using the outrageously simple Gradio framework.
#
# Prepare for joy!

# %%
# imports

import os
import requests
from bs4 import BeautifulSoup
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai
import anthropic

# %%
import gradio as gr  # oh yeah!

# %%
# Load environment variables in a file called .env

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-key-if-not-using-env")
os.environ["ANTHROPIC_API_KEY"] = os.getenv(
    "ANTHROPIC_API_KEY", "your-key-if-not-using-env"
)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "your-key-if-not-using-env")

# %%
# Connect to OpenAI, Anthropic and Google

openai = OpenAI()

claude = anthropic.Anthropic()

google.generativeai.configure()

# %%
# A generic system message - no more snarky adversarial AIs!

system_message = "You are a helpful assistant"

# %%
# Let's wrap a call to GPT-4o-mini in a simple function


def message_gpt(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return completion.choices[0].message.content


# %%
message_gpt("What is today's date?")

# %% [markdown]
# ## User Interface time!

# %%
# here's a simple function


def shout(text):
    print(f"Shout has been called with input {text}")
    return text.upper()


# %%
shout("hello")

# %%
gr.Interface(fn=shout, inputs="textbox", outputs="textbox").launch()

# %%
gr.Interface(
    fn=shout, inputs="textbox", outputs="textbox", flagging_mode="never"
).launch(share=True)

# %%
view = gr.Interface(
    fn=shout,
    inputs=[gr.Textbox(label="Your message:", lines=6)],
    outputs=[gr.Textbox(label="Response:", lines=8)],
    allow_flagging="never",
)
view.launch()

# %%
view = gr.Interface(
    fn=message_gpt,
    inputs=[gr.Textbox(label="Your message:", lines=6)],
    outputs=[gr.Textbox(label="Response:", lines=8)],
    allow_flagging="never",
)
view.launch()

# %%
system_message = "You are a helpful assistant that responds in markdown"

view = gr.Interface(
    fn=message_gpt,
    inputs=[gr.Textbox(label="Your message:")],
    outputs=[gr.Markdown(label="Response:")],
    allow_flagging="never",
)
view.launch()

# %%
# Let's create a call that streams back results


def stream_gpt(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    stream = openai.chat.completions.create(
        model="gpt-4o-mini", messages=messages, stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result


# %%
view = gr.Interface(
    fn=stream_gpt,
    inputs=[gr.Textbox(label="Your message:")],
    outputs=[gr.Markdown(label="Response:")],
    allow_flagging="never",
)
view.launch()


# %%
def stream_claude(prompt):
    result = claude.messages.stream(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0.7,
        system=system_message,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    response = ""
    with result as stream:
        for text in stream.text_stream:
            response += text or ""
            yield response


# %%
view = gr.Interface(
    fn=stream_claude,
    inputs=[gr.Textbox(label="Your message:")],
    outputs=[gr.Markdown(label="Response:")],
    allow_flagging="never",
)
view.launch()


# %%
def stream_model(prompt, model):
    if model == "GPT":
        result = stream_gpt(prompt)
    elif model == "Claude":
        result = stream_claude(prompt)
    else:
        raise ValueError("Unknown model")
    for chunk in result:
        yield chunk


# %%
view = gr.Interface(
    fn=stream_model,
    inputs=[
        gr.Textbox(label="Your message:"),
        gr.Dropdown(["GPT", "Claude"], label="Select model"),
    ],
    outputs=[gr.Markdown(label="Response:")],
    allow_flagging="never",
)
view.launch()

# %% [markdown]
# # Building a company brochure generator
#
# Now you know how - it's simple!

# %%
# A class to represent a Webpage


class Website:
    url: str
    title: str
    text: str

    def __init__(self, url):
        self.url = url
        response = requests.get(url)
        self.body = response.content
        soup = BeautifulSoup(self.body, "html.parser")
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"


# %%
system_prompt = "You are an assistant that analyzes the contents of a company website landing page \
and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown."


# %%
def stream_brochure(company_name, url, model):
    prompt = f"Please generate a company brochure for {company_name}. Here is their landing page:\n"
    prompt += Website(url).get_contents()
    if model == "GPT":
        result = stream_gpt(prompt)
    elif model == "Claude":
        result = stream_claude(prompt)
    else:
        raise ValueError("Unknown model")
    for chunk in result:
        yield chunk


# %%
view = gr.Interface(
    fn=stream_brochure,
    inputs=[
        gr.Textbox(label="Company name:"),
        gr.Textbox(label="Landing page URL:"),
        gr.Dropdown(["GPT", "Claude"], label="Select model"),
    ],
    outputs=[gr.Markdown(label="Brochure:")],
    allow_flagging="never",
)
view.launch()

# %%
