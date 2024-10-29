# %% [markdown]
# # Project - Airline AI Assistant
#
# We'll now bring together what we've learned to make an AI Customer Support assistant for an Airline

# %%
# imports

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

# %%
# Initialization

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-key-if-not-using-env")
MODEL = "gpt-4o-mini"
openai = OpenAI()

# %%
system_message = "You are a helpful assistant for an Airline called FlightAI. "
system_message += "Give short, courteous answers, no more than 1 sentence. "
system_message += "Always be accurate. If you don't know the answer, say so."


# %%
def chat(message, history):
    messages = [{"role": "system", "content": system_message}]
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content


gr.ChatInterface(fn=chat).launch()

# %% [markdown]
# ## Tools
#
# Tools are an incredibly powerful feature provided by the frontier LLMs.
#
# With tools, you can write a function, and have the LLM call that function as part of its response.
#
# Sounds almost spooky.. we're giving it the power to run code on our machine?
#
# Well, kinda.

# %%
# Let's start by making a useful function

ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}


def get_ticket_price(destination_city):
    print(f"Tool get_ticket_price called for {destination_city}")
    city = destination_city.lower()
    return ticket_prices.get(city, "Unknown")


# %%
get_ticket_price("London")

# %%
# There's a particular dictionary structure that's required to describe our function:

price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city. Call this whenever you need to know the ticket price, for example when a customer asks 'How much is a ticket to this city'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False,
    },
}

# %%
# And this is included in a list of tools:

tools = [{"type": "function", "function": price_function}]

# %% [markdown]
# ## Getting OpenAI to use our Tool
#
# There's some fiddly stuff to allow OpenAI "to call our tool"
#
# What we actually do is give the LLM the opportunity to inform us that it wants us to run the tool.
#
# Here's how the new chat function looks:


# %%
# We have to write that function handle_tool_call:
def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    city = arguments.get("destination_city")
    price = get_ticket_price(city)
    response = {
        "role": "tool",
        "content": json.dumps({"destination_city": city, "price": price}),
        "tool_call_id": message.tool_calls[0].id,
    }
    return response, city


def chat(message, history):
    messages = [{"role": "system", "content": system_message}]
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})
    response = openai.chat.completions.create(
        model=MODEL, messages=messages, tools=tools
    )

    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        response, city = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        response = openai.chat.completions.create(model=MODEL, messages=messages)

    return response.choices[0].message.content


# %%
gr.ChatInterface(fn=chat).launch()

# %% [markdown]
# # Let's go multi-modal!!
#
# We can use DALL-E-3, the image generation model behind GPT-4o, to make us some images
#
# Let's put this in a function called artist.
#
# ### Price alert: each time I generate an image it costs about 4c - don't go crazy with images!

# %%
# Some imports for handling images

import base64
from io import BytesIO
from PIL import Image


# %%
def artist(city):
    image_response = openai.images.generate(
        model="dall-e-3",
        prompt=f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style",
        size="1024x1024",
        n=1,
        response_format="b64_json",
    )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))


# %%
image = artist("New York City")
display(image)

# %%


# %% [markdown]
# ## Audio
#
# And let's make a function talker that uses OpenAI's speech model to generate Audio
#
# ### Troubleshooting Audio issues
#
# If you have any problems running this code below (like a FileNotFound error, or a warning of a missing package), you may need to install FFmpeg, a very popular audio utility.
#
# **For PC Users**
#
# 1. Download FFmpeg from the official website: https://ffmpeg.org/download.html
#
# 2. Extract the downloaded files to a location on your computer (e.g., `C:\ffmpeg`)
#
# 3. Add the FFmpeg bin folder to your system PATH:
# - Right-click on 'This PC' or 'My Computer' and select 'Properties'
# - Click on 'Advanced system settings'
# - Click on 'Environment Variables'
# - Under 'System variables', find and edit 'Path'
# - Add a new entry with the path to your FFmpeg bin folder (e.g., `C:\ffmpeg\bin`)
# - Restart your command prompt, and within Jupyter Lab do Kernel -> Restart kernel, to pick up the changes
#
# 4. Open a new command prompt and run this to make sure it's installed OK
# `ffmpeg -version`
#
# **For Mac Users**
#
# 1. Install homebrew if you don't have it already by running this in a Terminal window and following any instructions:
# `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
#
# 2. Then install FFmpeg with `brew install ffmpeg`
#
# 3. Verify your installation with `ffmpeg -version` and if everything is good, within Jupyter Lab do Kernel -> Restart kernel to pick up the changes
#
# Message me or email me at ed@edwarddonner.com with any problems!

# %%
from pydub import AudioSegment
from pydub.playback import play


def talker(message):
    response = openai.audio.speech.create(
        model="tts-1",
        voice="onyx",  # Also, try replacing onyx with alloy
        input=message,
    )

    audio_stream = BytesIO(response.content)
    audio = AudioSegment.from_file(audio_stream, format="mp3")
    play(audio)


# %%
talker("Well, hi there")

# %% [markdown]
# # Our Agent Framework
#
# The term 'Agentic AI' and Agentization is an umbrella term that refers to a number of techniques, such as:
#
# 1. Breaking a complex problem into smaller steps, with multiple LLMs carrying out specialized tasks
# 2. The ability for LLMs to use Tools to give them additional capabilities
# 3. The 'Agent Environment' which allows Agents to collaborate
# 4. An LLM can act as the Planner, dividing bigger tasks into smaller ones for the specialists
# 5. The concept of an Agent having autonomy / agency, beyond just responding to a prompt - such as Memory
#
# We're showing 1 and 2 here, and to a lesser extent 3 and 5. In week 8 we will do the lot!


# %%
def chat(message, history):
    image = None
    conversation = [{"role": "system", "content": system_message}]
    for human, assistant in history:
        conversation.append({"role": "user", "content": human})
        conversation.append({"role": "assistant", "content": assistant})
    conversation.append({"role": "user", "content": message})
    response = openai.chat.completions.create(
        model=MODEL, messages=conversation, tools=tools
    )

    if response.choices[0].finish_reason == "tool_calls":
        message = tool_call = response.choices[0].message
        response, city = handle_tool_call(message)
        conversation.append(message)
        conversation.append(response)
        image = artist(city)
        response = openai.chat.completions.create(model=MODEL, messages=conversation)

    reply = response.choices[0].message.content
    talker(reply)
    return reply, image


# %%
# More involved Gradio code as we're not using the preset Chat interface

with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500)
        image_output = gr.Image(height=500)
    with gr.Row():
        msg = gr.Textbox(label="Chat with our AI Assistant:")
    with gr.Row():
        clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        user_message = history[-1][0]
        bot_message, image = chat(user_message, history[:-1])
        history[-1][1] = bot_message
        return history, image

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, [chatbot, image_output]
    )
    clear.click(lambda: None, None, chatbot, queue=False)

ui.launch()

# %%
