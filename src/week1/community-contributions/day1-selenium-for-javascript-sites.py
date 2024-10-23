# %% [markdown]
# # Instant Gratification!
#
# Let's build a useful LLM solution - in a matter of minutes.
#
# Our goal is to code a new kind of Web Browser. Give it a URL, and it will respond with a summary. The Reader's Digest of the internet!!
#
# Before starting, be sure to have followed the instructions in the "README" file, including creating your API key with OpenAI and adding it to the `.env` file.
#
# ## If you're new to Jupyer Lab
#
# Welcome to the wonderful world of Data Science experimentation! Once you've used Jupyter Lab, you'll wonder how you ever lived without it. Simply click in each "cell" with code in it, like the cell immediately below this text, and hit Shift+Return to execute that cell. As you wish, you can add a cell with the + button in the toolbar, and print values of variables, or try out variations.
#
# If you need to start again, go to Kernel menu >> Restart kernel.

# %%
# imports

import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from openai import OpenAI

# %% [markdown]
# # Connecting to OpenAI
#
# The next cell is where we load in the environment variables in your `.env` file and connect to OpenAI.
#
# ## Troubleshooting if you have problems:
#
# 1. OpenAI takes a few minutes to register after you set up an account. If you receive an error about being over quota, try waiting a few minutes and try again.
# 2. Also, double check you have the right kind of API token with the right permissions. You should find it on [this webpage](https://platform.openai.com/api-keys) and it should show with Permissions of "All". If not, try creating another key by:
# - Pressing "Create new secret key" on the top right
# - Select **Owned by:** you, **Project:** Default project, **Permissions:** All
# - Click Create secret key, and use that new key in the code and the `.env` file (it might take a few minutes to activate)
# - Do a Kernel >> Restart kernel, and execute the cells in this Jupyter lab starting at the top
# 4. As a fallback, replace the line `openai = OpenAI()` with `openai = OpenAI(api_key="your-key-here")` - while it's not recommended to hard code tokens in Jupyter lab, because then you can't share your lab with others, it's a workaround for now
# 5. Contact me! Message me or email ed@edwarddonner.com and we will get this to work.
#
# Any concerns about API costs? See my notes in the README - costs should be minimal, and you can control it at every point.

# %%
# Load environment variables in a file called .env

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-key-if-not-using-env")
openai = OpenAI()

# %%
# A class to represent a Webpage


class Website:
    url: str
    title: str
    text: str

    def __init__(self, url):
        self.url = url
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)


# %%
# Let's try one out

ed = Website("https://edwarddonner.com")
print(ed.title)
print(ed.text)

# %% [markdown]
# ## Types of prompts
#
# You may know this already - but if not, you will get very familiar with it!
#
# Models like GPT4o have been trained to receive instructions in a particular way.
#
# They expect to receive:
#
# **A system prompt** that tells them what task they are performing and what tone they should use
#
# **A user prompt** -- the conversation starter that they should reply to

# %%
system_prompt = "You are an assistant that analyzes the contents of a website \
and provides a short summary, ignoring text that might be navigation related. \
Respond in markdown."


# %%
def user_prompt_for(website):
    user_prompt = f"You are looking at a website titled {website.title}"
    user_prompt += "The contents of this website is as follows; \
please provide a short summary of this website in markdown. \
If it includes news or announcements, then summarize these too.\n\n"
    user_prompt += website.text
    return user_prompt


# %% [markdown]
# ## Messages
#
# The API from OpenAI expects to receive messages in a particular structure.
# Many of the other APIs share this structure:
#
# ```
# [
#     {"role": "system", "content": "system message goes here"},
#     {"role": "user", "content": "user message goes here"}
# ]


# %%
def messages_for(website):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)},
    ]


# %% [markdown]
# ## Time to bring it together - the API for OpenAI is very simple!


# %%
def summarize(url):
    website = Website(url)
    response = openai.chat.completions.create(
        model="gpt-4o-mini", messages=messages_for(website)
    )
    return response.choices[0].message.content


# %%
summarize("https://edwarddonner.com")


# %%
def display_summary(url):
    summary = summarize(url)
    display(Markdown(summary))


# %%
display_summary("https://edwarddonner.com")

# %%
display_summary("https://cnn.com")

# %%
display_summary("https://anthropic.com")

# %% [markdown]
# ## An extra exercise for those who enjoy web scraping
#
# You may notice that if you try `display_summary("https://openai.com")` - it doesn't work! That's because OpenAI has a fancy website that uses Javascript. There are many ways around this that some of you might be familiar with. For example, Selenium is a hugely popular framework that runs a browser behind the scenes, renders the page, and allows you to query it. If you have experience with Selenium, Playwright or similar, then feel free to improve the Website class to use them. Please push your code afterwards so I can share it with other students!

# %%
display_summary("https://openai.com")

# %%
# Parse webpages which is designed using JavaScript heavely
# download the chorme driver from here as per your version of chrome - https://developer.chrome.com/docs/chromedriver/downloads
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

PATH_TO_CHROME_DRIVER = (
    "/home/voldemort/.wdm/drivers/chromedriver/linux64/115.0.5763.0/chromedriver"
)


class Website:
    url: str
    title: str
    text: str

    def __init__(self, url):
        self.url = url

        options = Options()

        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        service = Service(PATH_TO_CHROME_DRIVER)
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(url)

        input(
            "Please complete the verification in the browser and press Enter to continue..."
        )
        page_source = driver.page_source
        driver.quit()

        soup = BeautifulSoup(page_source, "html.parser")
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.get_text(separator="\n", strip=True)


# %%
display_summary("https://openai.com")

# %%
display_summary("https://edwarddonner.com")

# %%
display_summary("https://cnn.com")

# %%
