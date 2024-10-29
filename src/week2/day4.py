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
get_ticket_price("Berlin")

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
