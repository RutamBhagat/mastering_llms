# %% [markdown]
# # A full business solution
#
# Create a product that builds a Brochure for a company to be used for prospective clients, investors and potential recruits.
#
# We will be provided a company name and their primary website.
#
# See the end of this notebook for examples of real-world business applications.
#
# And remember: I'm always available if you have problems or ideas! Please do reach out.

# %%
# imports

import os
import requests
import json
from typing import List
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display, update_display
from openai import OpenAI

# %%
# Initialize and constants

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-key-if-not-using-env")
MODEL = "gpt-4o-mini"
openai = OpenAI()

# %%
# A class to represent a Webpage


class Website:
    url: str
    title: str
    body: str
    links: List[str]
    text: str

    def __init__(self, url):
        self.url = url
        response = requests.get(url)
        self.body = response.content
        soup = BeautifulSoup(self.body, "html.parser")
        self.title = soup.title.string if soup.title else "No title found"
        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""
        links = [link.get("href") for link in soup.find_all("a")]
        self.links = [link for link in links if link]

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"


# %%
ed = Website("https://edwarddonner.com")
print(ed.get_contents())

# %% [markdown]
# ## First step: Have GPT-4o-mini figure out which links are relevant
#
# ### Use a call to gpt-4o-mini to read the links on a webpage, and respond in structured JSON.
# It should decide which links are relevant, and replace relative links such as "/about" with "https://company.com/about".
# We will use "one shot prompting" in which we provide an example of how it should respond in the prompt.
#
# Sidenote: there is a more advanced technique called "Structured Outputs" in which we require the model to respond according to a spec. We cover this technique in Week 8 during our autonomous Agentic AI project.

# %%
link_system_prompt = "You are provided with a list of links found on a webpage. \
You are able to decide which of the links would be most relevant to include in a brochure about the company, \
such as links to an About page, or a Company page, or Careers/Jobs pages.\n"
link_system_prompt += "You should respond in JSON as in this example:"
link_system_prompt += """
{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page": "url": "https://another.full.url/careers"}
    ]
}
"""


# %%
def get_links_user_prompt(website):
    user_prompt = f"Here is the list of links on the website of {website.url} - "
    user_prompt += "please decide which of these are relevant web links for a brochure about the company, respond with the full https URL in JSON format. \
Do not include Terms of Service, Privacy, email links.\n"
    user_prompt += "Links (some might be relative links):\n"
    user_prompt += "\n".join(website.links)
    return user_prompt


# %%
print(get_links_user_prompt(ed))


# %%
def get_links(url):
    website = Website(url)
    completion = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_user_prompt(website)},
        ],
        response_format={"type": "json_object"},
    )
    result = completion.choices[0].message.content
    return json.loads(result)


# %%
get_links("https://anthropic.com")

# %% [markdown]
# ## Second step: make the brochure!
#
# Assemble all the details into another prompt to GPT4-o


# %%
def get_all_details(url):
    result = "Landing page:\n"
    result += Website(url).get_contents()
    links = get_links(url)
    print("Found links:", links)
    for link in links["links"]:
        result += f"\n\n{link['type']}\n"
        result += Website(link["url"]).get_contents()
    return result


# %%
print(get_all_details("https://anthropic.com"))

# %%
system_prompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
Include details of company culture, customers and careers/jobs if you have the information."

# Or uncomment the lines below for a more humorous brochure - this demonstrates how easy it is to incorporate 'tone':

# system_prompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
# and creates a short humorous, entertaining, jokey brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
# Include details of company culture, customers and careers/jobs if you have the information."


# %%
def get_brochure_user_prompt(company_name, url):
    user_prompt = f"You are looking at a company called: {company_name}\n"
    user_prompt += f"Here are the contents of its landing page and other relevant pages; use this information to build a short brochure of the company in markdown.\n"
    user_prompt += get_all_details(url)
    user_prompt = user_prompt[:20_000]  # Truncate if more than 20,000 characters
    return user_prompt


# %%
def create_brochure(company_name, url):
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)},
        ],
    )
    result = response.choices[0].message.content
    display(Markdown(result))


# %%
create_brochure("Anthropic", "https://anthropic.com")

# %% [markdown]
# ## Finally - a minor improvement
#
# With a small adjustment, we can change this so that the results stream back from OpenAI,
# with the familiar typewriter animation

# %%


# %%
def stream_brochure(company_name, url):
    stream = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)},
        ],
        stream=True,
    )

    response = ""
    display_handle = display(Markdown(""), display_id=True)
    for chunk in stream:
        response += chunk.choices[0].delta.content or ""
        response = response.replace("```", "").replace("markdown", "")
        update_display(Markdown(response), display_id=display_handle.display_id)


# %%
stream_brochure("Anthropic", "https://anthropic.com")

# %%
stream_brochure("HuggingFace", "https://huggingface.co")

# %% [markdown]
# ## Business Applications
#
# In this exercise we extended the Day 1 code to make multiple LLM calls, and generate a document.
#
# In terms of techniques, this is perhaps the first example of Agentic AI design patterns, as we combined multiple calls to LLMs. This will feature more in Week 2, and then we will return to Agentic AI in a big way in Week 8 when we build a fully autonomous Agent solution.
#
# In terms of business applications - generating content in this way is one of the very most common Use Cases. As with summarization, this can be applied to any business vertical. Write marketing content, generate a product tutorial from a spec, create personalized email content, and so much more. Explore how you can apply content generation to your business, and try making yourself a proof-of-concept prototype.

# %%
