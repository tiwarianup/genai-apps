from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = ""

llm = ChatOpenAI(
    model = "crewai-llama3",
    base_url = "http://localhost:11434/v1"
)

################# DEFINE AGENTS & TASKS ####################

# content planner agent
planner = Agent(
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory="You are working on planning a blog article "
              "about the topic: {topic} in 'https://medium.com/'. "
              "You collect infomation that helps the "
              "audience learn something "
              "and make informed decisions."
              "You have to prepare a detailed "
              "outline and the relevant topics and sub-topics that has to be part of the "
              "blogpost."
              "Your work is the basis for "
              "the Content Writer to write an article on this topic.",
    llm=llm,
    allow_delegation=True,
    verbose=True
)
