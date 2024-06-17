import uuid
import numpy as np, pandas as pd
from .prompts import extractConcepts
from .prompts import graphPrompt

def documents2Dataframe(documents) -> pd.DataFrame:
    rows = []
    for chunk in documents:
        row = {
            "text": chunk.page_content,
            **chunk.metadata,
            "chunk_id": uuid.uuid4().hex,
        }
        rows = rows + [row]
    
    df = pd.DataFrame(rows)
    return df

def df2ConceptsList(dataframe: pd.DataFrame) -> list:
    results = dataframe.apply(
        lambda row: extractConcepts(
            row.text, {"chunk_id": row.chunk_id, "type": "concept"}
        ),
        axis = 1
    )
    
    #invalid JSON results handling
    results = results.dropna()
    results = results.reset_index(drop=True)
    
    #flatten the list of lists
    concepts_list = np.concatenate(results).ravel.tolist()
    
    return concepts_list

def concept2Df(concepts_list) -> pd.DataFrame:
    #remove all NaN entries
    concepts_dataframe = pd.DataFrame(concepts_list).replace(" ", np.nan)
    concepts_dataframe = concepts_dataframe.dropna(subset=["entity"])
    concepts_dataframe["entity"] = concepts_dataframe["entity"].apply(lambda x: x.lower())
    
    return concepts_dataframe

def df2Graph(dataframe: pd.DataFrame, model=None) -> list:
    results = dataframe.apply(
        lambda row: graphPrompt(row.text, {"chunk_id": row.chunk_id}, model), axis=1
    )
    
    results = results.dropna()
    results = results.reset_index(drop=True)
    
    #flatted the list of list
    concepts_list = np.concatenate(results).ravel().tolist()
    
    return concepts_list

def graph2Df(nodes_list) -> pd.DataFrame:
    graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)
    graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])
    graph_dataframe["node_1"] = graph_dataframe["node_1"].apply(lambda x: x.lower())
    graph_dataframe["node_2"] = graph_dataframe["node_2"].apply(lambda x: x.lower())
    
    return graph_dataframe
    