import numpy as np, pandas as pd
import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader, PyPDFium2Loader
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pathlib import Path
import random

print(os.getcwd()) 

inp_data_dir = Path(f"./data_input/")
out_data_dir = Path(f"./data_output/")

loader = DirectoryLoader(inp_data_dir, show_progress=True)
documents = loader.load()

# print(documents)
 
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150,
    length_function = len,
    is_separator_regex = False
)

pages = splitter.split_documents(documents)
print(pages[3].page_content[:100])

########### EXTRACT CONCEPTS ###############
from helpers.df_helpers import documents2Dataframe

df = documents2Dataframe(pages)
print(df.shape)
print(df.head(2))


from helpers.df_helpers import df2Graph
from helpers.df_helpers import graph2Df

regenerate = True

if regenerate:
    concepts_list = df2Graph(df, model="zephyr:latest")
    dfgraph1 = graph2Df(concepts_list)
    
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)
    
    print(dfgraph1.head())
    dfgraph1.to_csv(out_data_dir/'graph.csv', sep="|", index=False)
    
    print(df.head())
    df.to_csv(out_data_dir/'chunks.csv', sep="|", index=False)
    
else:
    dfgraph1 = pd.read_csv(out_data_dir/'graph.csv', sep="|")

dfgraph1.replace("", np.nan, inplace=True)
dfgraph1.dropna(subset=["node_1", "node_2", "edge"], inplace=True)
dfgraph1['count'] = 4

print(dfgraph1.shape)
print(dfgraph1.head())

############### CALCULATE CONTEXTUAL PROXIMITY ################


def contextual_proximity(df: pd.DataFrame) -> pd.DataFrame:
    # melt the dataframe into a list of nodes
    
    dfg_long = pd.melt(
        df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node"
    )
    
    dfg_long.drop(columns=["variable"], inplace=True)
    
    #self join with chunk_id to create a link between the terms occuring in the same text chunk
    dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))
    
    #drop self-loops in the graph
    self_loops_drop = dfg_wide[dfg_wide["node_1"] == dfg_wide["node_2"]].index
    dfgraph2 = dfg_wide.drop(index=self_loops_drop).reset_index(drop=True)
    
    #group and count the edges
    dfgraph2 = (
        dfgraph2.groupby(["node_1", "node_2"])
        .agg({"chunk_id": [",".join, "count"]})
        .reset_index()
    )
    
    dfgraph2.columns = ["node_1", "node_2", "chunk_id", "count"]
    dfgraph2.replace("", np.nan, inplace=True)
    dfgraph2.dropna(subset=["node_1", "node_2"], inplace=True)
    
    #drop edges with a 1 count
    dfgraph2 = dfgraph2[dfgraph2["count"] != 1] 
    dfgraph2["edge"] = "contextual proximity"
    
    return dfgraph2

dfgraph2 = contextual_proximity(dfgraph1)
dfgraph2.tail()

# merge both dataframes
dfg = pd.concat([dfgraph1, dfgraph2], axis=0)

dfg = dfg.groupby(["node_1", "node_2"]).agg(
    {
        "chunk_id": ",".join, 
        "edge": ",".join, 
        "count": "sum"
     }
).reset_index()

print(dfg.head())

#NetworkX graph
nodes = pd.concat([dfg["node_1"], dfg["node_2"]], axis=0).unique()
print(nodes.shape)

import networkx as nx

G = nx.Graph()

#add nodes to the graph
for node in nodes:
    G.add_node(
        str(node)
    )

#add edges to the graph
for index, row in dfg.iterrows():
    G.add_edge(
        str(row["node_1"]),
        str(row["node_2"]),
        title = row["edge"],
        weight = row["count"]/4
    )

#calculate communities for coloring the node
communities_generator = nx.community.girvan_newman(G)
top_level_communities = next(communities_generator)
next_level_communities = next(communities_generator)
communities = sorted(map(sorted, next_level_communities))
print("No of communties: ", len(communities))
print(communities)

#create a df for community colors
import seaborn as sns
palette = "hls"

#add color to communities and make another dataframe
def colors2community(communities) -> pd.DataFrame:
    
    p = sns.color_palette(palette, len(communities)).as_hex()
    random.shuffle(p)
    rows = []
    group = 0
    
    for community in communities:
        color = p.pop()
        group += 1
        for node in community:
            rows += [{"node": node, "color": color, "group": group}]
    
    df_colors = pd.DataFrame(rows)
    return df_colors

colors = colors2community(communities)
print(colors.head())

#add colors to the graph 
for index, row in colors.iterrows():
    G.nodes[row['node']]['group'] = row['group']
    G.nodes[row['node']]['color'] = row['color']
    G.nodes[row['node']]['size'] = G.degree[row['node']]
    

#visualize the network graph using pyvis

from pyvis.network import Network

graph_output_directory = "./docs/index.html"

net = Network(
    notebook = False,
    cdn_resources = "remote",
    height = "900px",
    width = "100%",
    select_menu = True,
    filter_menu = True
)

net.from_nx(G)
net.force_atlas_2based(central_gravity=0.015, gravity=-31)
net.show_buttons(filter_=['transmission'])

net.show(graph_output_directory, notebook=False)