from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

import os
import json
import requests
from bs4 import BeautifulSoup

from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.utilities import DuckDuckGoSearchAPIWrapper


RESULTS_PER_QUESTION = 3

ddg_search = DuckDuckGoSearchAPIWrapper()


os.environ["OPENAI_API_KEY"] = ""



def web_search(query:str, num_results:int = RESULTS_PER_QUESTION):
    results = ddg_search.results(query, num_results)
    return [r["link"] for r in results]


SUMMARY_TEMPLATE = """{text}
-------------
Using the above text, answer the following questions briefly:
> {question}
if the question cannot be answered using the text, simply summarize the text.
Include all factual information, numbers, stats etc. if available.
"""

SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)

def scrape_text(url: str):
    #Send a GET request to the webpage
    try:
        response = requests.get(url)
        
        #Check if the request was successful
        if response.status_code == 200:
            #Parse the content of the request with BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract all the text from the webpage
            page_text = soup.get_text(separator=" ", strip=True)
            
            # Print the extracted text
            return page_text
        else:
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
    except Exception as e:
        print(e)
        return f"Failed to retreive the webpage: {e}"
    
url = "https://blog.langchain.dev/announcing-langsmith/"

scrape_and_summarize_chain = RunnablePassthrough.assign(
    summary = RunnablePassthrough.assign(
        text = lambda x: scrape_text(x["url"])[:20000]
    ) | SUMMARY_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()
) | (lambda x: f"URL: {x['url']}\n\nSUMMARY: {x['summary']}")

web_search_chain = RunnablePassthrough.assign(
    urls = lambda x: web_search(x["question"])
) | (lambda x:  [{"question": x["question"], "url": u} for u in x["urls"]]) | scrape_and_summarize_chain.map()


SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "Write 3 search queries to search online that form an "
            "objective opinion from the following: {question}\n"
            "You must respond with a list of strings in the following format: "
            '["query 1", "query 2", "query 3"].'
        ),
    ]
)

search_question_chain = SEARCH_PROMPT | ChatOpenAI(temperature=0) | StrOutputParser() | json.loads
full_research_chain = search_question_chain | (lambda x: [{"question": q} for q in x]) | web_search_chain.map()

WRITER_SYSTEM_PROMPT = "You are an AI critical thinker research assistant.Your sole purpose is to write well written, critically acclaimed, objective and structured reports on the given text."

RESEARCH_REPORT_TEMPLATE  = """Information:
------------
{research_summary}
------------
Using the above information, answer the following question or topic: "{question}" in a detailed report --\
The report should focus on the answer to the question, should be well structured, informative, \
in depth, with facts and numbers if available and a minimum of 1200 words.
You should strive to write the report as long as you can using all relevant and necessary information provided.
You must write the report with markdown syntax.
You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusion.
Write all used source urls at the end if the report, and make sure to not add duplicated sources, but only one reference fr each.
You must write the report in apa format.
Please do your best, this is very important to my career."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WRITER_SYSTEM_PROMPT),
        ("user", RESEARCH_REPORT_TEMPLATE),
    ]
)

def collapse_list_of_lists(list_of_lists):
    content = []
    for lst in list_of_lists:
        content.append("\n\n".join(lst))
    return "\n\n".join(content)

chain = RunnablePassthrough.assign(
    research_summary = full_research_chain | collapse_list_of_lists
) | prompt | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()

async def get_research_report(question: str):
    report_data = chain.run(question=question)
    modified_data = RunnablePassthrough.assign(
    summary=report_data["summary"],  # Assign existing value from report_data
    analysis=lambda x: x["report"] * 2  # Function to manipulate another value
)(report_data)
    return modified_data

#!/usr/bin/env python
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
# from starlette.templates import TemplateDir
from langserve import add_routes

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces"
)

add_routes(
    app,
    chain, 
    path="/research-assistant",
)

# @app.get("/research-assistant")
# async def research_assistant(question: str):
#     report = await get_research_report(question)
#     return {"report": report}  # Return the report in a dictionary

# templates = Jinja2Templates(directory="templates")  # Assuming templates directory

# @app.get("/research-assistant")
# async def research_assistant_ui():
#     question = ""  # Set an initial empty question value
#     report_data = chain.invoke(question)
#     return templates.TemplateResponse("research_assistant.html", context=modified_data)

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app=app, host="localhost", port=8000)


