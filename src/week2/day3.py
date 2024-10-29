# %% [markdown]
# # Day 3 - Conversational AI - aka Chatbot!

# %%
# imports

import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

# %%
# Load environment variables in a file called .env

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-key-if-not-using-env")
os.environ["ANTHROPIC_API_KEY"] = os.getenv(
    "ANTHROPIC_API_KEY", "your-key-if-not-using-env"
)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "your-key-if-not-using-env")

# %%
# Initialize

openai = OpenAI()
MODEL = "gpt-4o-mini"

# %%
system_message = "You are a helpful assistant"

# %% [markdown]
# ## Reminder of the structure of prompt messages to OpenAI:
#
# ```
# [
#     {"role": "system", "content": "system message here"},
#     {"role": "user", "content": "first user prompt here"},
#     {"role": "assistant", "content": "the assistant's response"},
#     {"role": "user", "content": "the new user prompt"},
# ]
# ```
#
# We will write a function `chat(message, history)` where:
# **message** is the prompt to use
# **history** is a list of pairs of user message with assistant's reply
#
# ```
# [
#     ["user said this", "assistant replied"],
#     ["then user said this", "and assistant replied again],
#     ...
# ]
# ```
# We will convert this history into the prompt style for OpenAI, then call OpenAI.


# %%
def chat(message, history):
    messages = [{"role": "system", "content": system_message}]
    for user_message, assistant_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})
    messages.append({"role": "user", "content": message})

    print("History is:")
    print(history)
    print("And messages is:")
    print(messages)

    stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ""
        yield response


# %% [markdown]
# ## And then enter Gradio's magic!

# %%
gr.ChatInterface(fn=chat).launch()

# %%
system_message = "You are a helpful assistant in a clothes store. You should try to gently encourage \
the customer to try items that are on sale. Hats are 60% off, and most other items are 50% off. \
For example, if the customer says 'I'm looking to buy a hat', \
you could reply something like, 'Wonderful - we have lots of hats - including several that are part of our sales evemt.'\
Encourage the customer to buy hats if they are unsure what to get."


# %%
def chat(message, history):
    messages = [{"role": "system", "content": system_message}]
    for user_message, assistant_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})
    messages.append({"role": "user", "content": message})

    stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ""
        yield response


# %%
gr.ChatInterface(fn=chat).launch()

# %%
system_message += "\nIf the customer asks for shoes, you should respond that shoes are not on sale today, \
but remind the customer to look at hats!"

# %%
gr.ChatInterface(fn=chat).launch()


# %%
def chat(message, history):
    messages = [{"role": "system", "content": system_message}]
    for user_message, assistant_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})

    if "belt" in message:
        messages.append(
            {
                "role": "system",
                "content": "For added context, the store does not sell belts, \
but be sure to point out other items on sale",
            }
        )

    messages.append({"role": "user", "content": message})

    stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ""
        yield response


# %%
gr.ChatInterface(fn=chat).launch()

# %%
