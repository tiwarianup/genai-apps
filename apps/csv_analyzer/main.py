import google.generativeai as genai
import pandas as pd
import json
import typing_extensions
import streamlit as st

#gemini setup
genai.configure(api_key="AIzaSyAwoHa-8FBE3iq2HZqhmOyFA6B3JnGPlTI")

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

#defining pandas and response prompts
model_pandas = genai.GenerativeModel('gemini-1.5-flash-latest', system_instruction="You are an expert Python Developer who works with Pandas. You make sure to generate simple pandas 'command' for the user queries in JSON format. No need to add 'print' function. Analyze the data types of the columns before generating the command. If unfeasible, return 'None'.")

model_response = genai.GenerativeModel('gemini-1.5-flash-latest', system_instruction="Your task is to comprehend. You must analyze the user query and response data to generate a response data in natural language.")

#response schema
class Command(typing_extensions.TypedDict):
    command: str
    
#streamlit for UI

st.title("Structured/Tabular Data Analyst")
st.write("Backed by Gemini & Pandas")
st.write("Analyze your spreadsheets in natural language with Gemini flash.")

#add file
uploaded_csv = st.file_uploader("Choose a Excel/CSV file to upload and query")
if uploaded_csv is not None:
    try:
        df = pd.read_csv(uploaded_csv)
    except:
        df = pd.read_excel(uploaded_csv)
    
    head = str(df.head().to_dict())
    desc = str(df.describe().to_dict())
    cols = str(df.columns.tolist())
    dtype = str(df.dtypes.to_dict())
    
    #user query
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if user_query := st.chat_input():
        print(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

    
    final_query = f"The dataframe name is 'df'. df has the columns {cols} and thier datatypes are {dtype}. df is in the following format: {desc}. The head of df is {head}. You cannot use df.info() or any command that cannot be printed. Do not use a plot 'command'. Only use 'command' which can be printed in UI. Write a pandas 'command' for this query on the dataframe df: {user_query}"
    
    with st.spinner("Analyzing the data..."):
        response = model_pandas.generate_content(
            final_query,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                # response_schema=Command,
                temperature=0.3
            ),
            safety_settings= safety_settings
        )
        print("################## Reached here! ##########")
        command = json.loads(response.text)["command"]
        
        print(command)
        
    
    try:
        exec(f"data = {command}")
        natural_response = f"The final user query is {final_query}. The output of the command is {str(data)}. If the data is 'None', you can say 'Please ask a query to get started'. Do not mention the command used. Generate the response in natural language for the output."
        
        bot_response = model_response.generate_content(
            natural_response,
            generation_config=genai.GenerationConfig(temperature=0.7),
            safety_settings= safety_settings
        )
        st.chat_message("assistant").write(bot_response.text)
        st.session_state.messages.append({"role": "assistant", "content": bot_response.text})
        
    except Exception as e:
        st.session_state.messages.append({"role": "assistant", "content": "Error"})
        
        
# streamlit run app.py --server.enableXsrfProtection false