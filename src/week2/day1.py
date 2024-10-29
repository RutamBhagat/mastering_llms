# %% [markdown]
# # Welcome to Week 2!
#
# ## Frontier Model APIs
#
# In Week 1, we used multiple Frontier LLMs through their Chat UI, and we connected with the OpenAI's API.
#
# Today we'll connect with the APIs for Anthropic and Google, as well as OpenAI.

# %% [markdown]
# ## Setting up your keys
#
# If you haven't done so already, you'll need to create API keys from OpenAI, Anthropic and Google.
#
# For OpenAI, visit https://openai.com/api/
# For Anthropic, visit https://console.anthropic.com/
# For Google, visit https://ai.google.dev/gemini-api
#
# When you get your API keys, you need to set them as environment variables.
#
# EITHER (recommended) create a file called `.env` in this project root directory, and set your keys there:
#
# ```
# OPENAI_API_KEY=xxxx
# ANTHROPIC_API_KEY=xxxx
# GOOGLE_API_KEY=xxxx
# ```
#
# OR enter the keys directly in the cells below.

# %%
# imports

import os
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai
import anthropic
from IPython.display import Markdown, display, update_display

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
# All 3 APIs are similar
# Having problems with API files? You can use openai = OpenAI(api_key="your-key-here") and same for claude
# Having problems with Google Gemini setup? Then just skip Gemini; you'll get all the experience you need from GPT and Claude.

openai = OpenAI()

claude = anthropic.Anthropic()

google.generativeai.configure()

# %% [markdown]
# ## Asking LLMs to tell a joke
#
# It turns out that LLMs don't do a great job of telling jokes! Let's compare a few models.
# Later we will be putting LLMs to better use!
#
# ### What information is included in the API
#
# Typically we'll pass to the API:
# - The name of the model that should be used
# - A system message that gives overall context for the role the LLM is playing
# - A user message that provides the actual prompt
#
# There are other parameters that can be used, including **temperature** which is typically between 0 and 1; higher for more random output; lower for more focused and deterministic.

# %%
system_message = "You are an assistant that is great at telling jokes"
user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"

# %%
prompts = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt},
]

# %%
# GPT-3.5-Turbo

completion = openai.chat.completions.create(model="gpt-3.5-turbo", messages=prompts)
print(completion.choices[0].message.content)

# %%
# GPT-4o-mini
# Temperature setting controls creativity

completion = openai.chat.completions.create(
    model="gpt-4o-mini", messages=prompts, temperature=0.7
)
print(completion.choices[0].message.content)

# %%
# GPT-4o

completion = openai.chat.completions.create(
    model="gpt-4o", messages=prompts, temperature=0.4
)
print(completion.choices[0].message.content)

# %%
# Claude 3.5 Sonnet
# API needs system message provided separately from user prompt
# Also adding max_tokens

message = claude.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=200,
    temperature=0.7,
    system=system_message,
    messages=[
        {"role": "user", "content": user_prompt},
    ],
)

print(message.content[0].text)

# %%
# Claude 3.5 Sonnet again
# Now let's add in streaming back results

result = claude.messages.stream(
    model="claude-3-5-sonnet-20240620",
    max_tokens=200,
    temperature=0.7,
    system=system_message,
    messages=[
        {"role": "user", "content": user_prompt},
    ],
)

with result as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# %%
# The API for Gemini has a slightly different structure

gemini = google.generativeai.GenerativeModel(
    model_name="gemini-1.5-flash", system_instruction=system_message
)
response = gemini.generate_content(user_prompt)
print(response.text)

# %%
# To be serious! GPT-4o-mini with the original question

prompts = [
    {"role": "system", "content": "You are a helpful assistant"},
    {
        "role": "user",
        "content": "How do I decide if a business problem is suitable for an LLM solution?",
    },
]

# %%
# Have it stream back results in markdown

stream = openai.chat.completions.create(
    model="gpt-4o", messages=prompts, temperature=0.7, stream=True
)

reply = ""
display_handle = display(Markdown(""), display_id=True)
for chunk in stream:
    reply += chunk.choices[0].delta.content or ""
    reply = reply.replace("```", "").replace("markdown", "")
    update_display(Markdown(reply), display_id=display_handle.display_id)

# %% [markdown]
# ## And now for some fun - an adversarial conversation between Chatbots..
#
# You're already familar with prompts being organized into lists like:
#
# ```
# [
#     {"role": "system", "content": "system message here"},
#     {"role": "user", "content": "user prompt here"}
# ]
# ```
#
# In fact this structure can be used to reflect a longer conversation history:
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
# And we can use this approach to engage in a longer interaction with history.

# %%
# Let's make a conversation between GPT-4o-mini and Claude-3-haiku
# We're using cheap versions of models so the costs will be minimal

gpt_model = "gpt-4o-mini"
claude_model = "claude-3-haiku-20240307"

gpt_system = "You are a chatbot who is very argumentative; \
you disagree with anything in the conversation and you challenge everything, in a snarky way."

claude_system = "You are a very polite, courteous chatbot. You try to agree with \
everything the other person says, or find common ground. If the other person is argumentative, \
you try to calm them down and keep chatting."

gpt_messages = ["Hi there"]
claude_messages = ["Hi"]


# %%
def call_gpt():
    messages = [{"role": "system", "content": gpt_system}]
    for gpt, claude in zip(gpt_messages, claude_messages):
        messages.append({"role": "assistant", "content": gpt})
        messages.append({"role": "user", "content": claude})
    completion = openai.chat.completions.create(model=gpt_model, messages=messages)
    return completion.choices[0].message.content


# %%
call_gpt()


# %%
def call_claude():
    messages = []
    for gpt, claude_message in zip(gpt_messages, claude_messages):
        messages.append({"role": "user", "content": gpt})
        messages.append({"role": "assistant", "content": claude_message})
    messages.append({"role": "user", "content": gpt_messages[-1]})
    message = claude.messages.create(
        model=claude_model, system=claude_system, messages=messages, max_tokens=500
    )
    return message.content[0].text


# %%
call_claude()

# %%
call_gpt()

# %%
gpt_messages = ["Hi there"]
claude_messages = ["Hi"]

print(f"GPT:\n{gpt_messages[0]}\n")
print(f"Claude:\n{claude_messages[0]}\n")

for i in range(5):
    gpt_next = call_gpt()
    print(f"GPT:\n{gpt_next}\n")
    gpt_messages.append(gpt_next)

    claude_next = call_claude()
    print(f"Claude:\n{claude_next}\n")
    claude_messages.append(claude_next)

# %%
