#from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

from langchain.utilities import DuckDuckGoSearchAPIWrapper

import os
import requests
import openai
from dotenv import find_dotenv, load_dotenv
from bs4 import BeautifulSoup

load_dotenv(find_dotenv())

# openai.api_key = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = ""

RESULTS_PER_QUESTION = 3

ddg_search = DuckDuckGoSearchAPIWrapper()

def web_search(query: str, num_results:int = RESULTS_PER_QUESTION):
    results = ddg_search.results(query, num_results)
    return [r["link"] for r in results]




template = """Summarise the following question based on the context:

Question: {question}

Context: 
{context}

"""

SUMMARY_TEMPLATE = """{text} 
-----------
Using the above text, answer in short the following question: 
> {question}
-----------
if the question cannot be answered using the text, simply summarize the text. Include all factual information, numbers, stats etc if available."""  # noqa: E501


prompt = ChatPromptTemplate.from_template(template)
summary_prompt = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)

# url = "https://stackoverflow.com/questions/76981304/how-can-i-print-the-output-of-my-langchain-agent-easily" 
url = "https://blog.langchain.dev/announcing-langsmith/"


def scrape_text(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            page_text = soup.get_text(separator=" ", strip=True)
            return page_text
        else:
            return f"Failed to retreive the webpage: {url} with status code: {response.status_code}."
    except Exception as e:
        print(e)
        return f"Failed to retreive the webpage: {url}"

scrape_and_summarize_chain = RunnablePassthrough.assign(
    text=lambda x: scrape_text(x["url"])[:10000]
    ) | summary_prompt | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()

chain = RunnablePassthrough.assign(
    urls = lambda x: web_search(x["question"])
) | (lambda x: [{"question": x["question"], "url": u} for u in x["urls"]]) | scrape_and_summarize_chain.map()


print("\n")
print(chain.invoke(
    {
        "question": "What is so great about langsmith?",
    }
))

