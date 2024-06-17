import sys
from yachalk import chalk
sys.path.append('..')

import json
import ollama.client as client

def extractConcepts(prompt: str, metadata = {}, model="mistral-openarca:latest"):
    SYS_PROMPT = (
        "Your task is to extract the key concepts (and non personal entities) mentioned in the given context. "
        "Extract only the most important and atomistic concepts, if needed break the concept down into simpler concepts."
        "Categorize the concept into one of the following categories: "
        "[event, concept, place, object, document, organisation, condition, misc]\n"
        "Format your output as a list of JSON in the following format:\n"
        "[\n"
        "   {\n"
        '   "Entity" : The Concept, \n'
        '   "Importance" : The contextual importance of the concept on a scale of 1-5 (5 being the highest), \n'
        '   "Category": The type of Concept, \n'
        "   }, \n"
        "{ }, \n"
        "]\n"
    )
    
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=prompt)
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR: Here is the buggy response: ", response, "\n\n")
        result = None
    
    return result

def graphPrompt(input: str, metadata={}, model="mistral-openorca:latest"):
    if model == None:
        model = "mistral-openorca:latest"
    
    # model_info = client.show(model_name=model)
    # print(chalk.blue(model_info))
    
    SYS_PROMPT = (
        "You are a network graph maker who extracts terms and thier relations from a given context."
        "You are provided with the context chunk (delimited by ```). Your task is to extract the ontology "
        "of terms mentioned in the given context. These terms should represent key concepts as per the context. \n"
        "Thought 1: While traversing through each sentence, think about the key terms mentioned in it. \n"
            "\tTerms may include object, entity, location, organization, person, \n"
            "\tcondition, acronym, document, service, concept, etc."
            "\tTerms should be atomistic as possible\n\n"
        "Thought 2: Think about how these terms can have a one to one relationship with other terms. \n"
            "\tTerms that are mentioned in the same sentence or same paragraph are typically related to each other. \n"
            "\tTerms can be related to many other terms \n\n"
        "Thought 3: Find out the relation between each such related pair of terms"
        "and the relation between them, like the following: \n"
        "[\n"
        "   {\n"
        '   "node_1" : "A concept from extracted ontology", \n'
        '   "node_2" : "A related concept from extracted ontology", \n'
        '   "edge": "Relationship between the two concepts, node_1 and node_2 in one or two sentences," \n'
        "   }, {...} \n"
        "]"
        
    )
    
    USER_PROMPT = f"context: ```{input}``` \n\n output: "
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT)
    
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for iterm in result]
    except:
        print("\n\nERROR: Here is the buggy response: ", response, "\n\n")
        result = None
    
    return result