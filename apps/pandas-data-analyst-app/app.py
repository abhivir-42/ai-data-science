# AI Data Science - Pandas Data Analyst App
# -----------------------

# This app is designed to help you analyze data and create data visualisations from natural language requests.

# Imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from openai import OpenAI

import streamlit as st
import pandas as pd
import plotly.io as pio
import json

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI

from src.multiagents import PandasDataAnalyst
from src.agents import DataWranglingAgent, DataVisualisationAgent

# Note: We need to add these agents to work properly
# For now, we'll create a simplified version

# * APP INPUTS ----

MODEL_LIST = ["gpt-4o-mini", "gpt-4o"]
TITLE = "Pandas Data Analyst AI Copilot"

# ---------------------------
# Streamlit App Configuration
# ---------------------------

st.set_page_config(
    page_title=TITLE,
    page_icon="📊",
)
st.title(TITLE)

st.markdown("""
Welcome to the Pandas Data Analyst AI. Upload a CSV or Excel file and ask questions about the data.  
The AI agent will analyse your dataset and return either data tables or interactive charts.
""")

with st.expander("Example Questions", expanded=False):
    st.write(
        """
        ##### Sample Questions:
        
        -  Show the top 5 records by a specific column.
        -  Show the top 5 records by sales in a bar chart.
        -  Show the distribution of a categorical variable in a pie chart.
        -  Make a plot of sales over time for each category. Use colour to identify the categories.
        -  Calculate summary statistics for numeric columns.
        -  Show correlation between numeric variables.
        """
    )

# ---------------------------
# OpenAI API Key Entry and Test
# ---------------------------

st.sidebar.header("Enter your OpenAI API Key")

st.session_state["OPENAI_API_KEY"] = st.sidebar.text_input(
    "API Key",
    type="password",
    help="Your OpenAI API key is required for the app to function.",
)

# Test OpenAI API Key
if st.session_state["OPENAI_API_KEY"]:
    # Set the API key for OpenAI
    client = OpenAI(api_key=st.session_state["OPENAI_API_KEY"])

    # Test the API key (optional)
    try:
        # Example: Fetch models to validate the key
        models = client.models.list()
        st.success("API Key is valid!")
    except Exception as e:
        st.error(f"Invalid API Key: {e}")
        st.stop()
else:
    st.info("Please enter your OpenAI API Key to proceed.")
    st.stop()

# * OpenAI Model Selection
model_option = st.sidebar.selectbox("Choose OpenAI model", MODEL_LIST, index=0)
llm = ChatOpenAI(model=model_option, api_key=st.session_state["OPENAI_API_KEY"])

# ---------------------------
# File Upload and Data Preview
# ---------------------------

st.markdown("""
Upload a CSV or Excel file and ask questions about your data.  
The AI agent will analyse your dataset and return either data tables or interactive charts.
""")

# Demo data option
use_demo_data = st.checkbox("Use demo data", value=False)

if use_demo_data:
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/business-science/ai-data-science-team/refs/heads/master/data/churn_data.csv")
        st.subheader("Data Preview (Demo Data)")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error loading demo data: {e}")
        st.stop()
else:
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file", type=["csv", "xlsx", "xls"]
    )
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("Data Preview")
        st.dataframe(df.head())
    else:
        st.info("Please upload a CSV or Excel file or use demo data to get started.")
        st.stop()

# ---------------------------
# Initialize Chat Message History and Storage
# ---------------------------

msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you analyse your data?")

if "plots" not in st.session_state:
    st.session_state.plots = []

if "dataframes" not in st.session_state:
    st.session_state.dataframes = []

def display_chat_history():
    for msg in msgs.messages:
        with st.chat_message(msg.type):
            if "PLOT_INDEX:" in msg.content:
                plot_index = int(msg.content.split("PLOT_INDEX:")[1])
                st.plotly_chart(
                    st.session_state.plots[plot_index], key=f"history_plot_{plot_index}"
                )
            elif "DATAFRAME_INDEX:" in msg.content:
                df_index = int(msg.content.split("DATAFRAME_INDEX:")[1])
                st.dataframe(
                    st.session_state.dataframes[df_index],
                    key=f"history_dataframe_{df_index}",
                )
            else:
                st.write(msg.content)

# Render current messages from StreamlitChatMessageHistory
display_chat_history()

# ---------------------------
# Simplified Data Analysis Function
# ---------------------------

def simple_data_analysis(question: str, data: pd.DataFrame) -> dict:
    """
    Simplified data analysis function.
    For now, we'll implement basic functionality until we have all agents working.
    """
    question_lower = question.lower()
    
    # Basic data exploration
    if any(word in question_lower for word in ['describe', 'summary', 'statistics']):
        result_df = data.describe()
        return {
            "type": "table",
            "data": result_df,
            "message": "Here are the summary statistics for your dataset:"
        }
    
    elif any(word in question_lower for word in ['head', 'first', 'top']) and any(word in question_lower for word in ['5', 'five', '10', 'ten']):
        # Extract number
        if '5' in question_lower or 'five' in question_lower:
            n = 5
        elif '10' in question_lower or 'ten' in question_lower:
            n = 10
        else:
            n = 5
        
        result_df = data.head(n)
        return {
            "type": "table", 
            "data": result_df,
            "message": f"Here are the first {n} rows of your dataset:"
        }
    
    elif any(word in question_lower for word in ['correlation', 'corr']):
        # Show correlation matrix for numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            return {
                "type": "table",
                "data": corr_matrix,
                "message": "Here is the correlation matrix for numeric columns:"
            }
        else:
            return {
                "type": "message",
                "message": "Not enough numeric columns to calculate correlation."
            }
    
    elif any(word in question_lower for word in ['info', 'information', 'columns']):
        # Data info
        info_data = {
            'Column': data.columns,
            'Type': [str(dtype) for dtype in data.dtypes],
            'Non-Null Count': [data[col].count() for col in data.columns],
            'Null Count': [data[col].isnull().sum() for col in data.columns]
        }
        result_df = pd.DataFrame(info_data)
        return {
            "type": "table",
            "data": result_df,
            "message": "Here is information about your dataset columns:"
        }
    
    else:
        # Default response with basic info
        result_df = data.head()
        return {
            "type": "table",
            "data": result_df,
            "message": "Here's a preview of your data. Try asking for 'summary statistics', 'correlation', or 'column information'."
        }

# ---------------------------
# Chat Input and Processing
# ---------------------------

if question := st.chat_input("Enter your question here:", key="query_input"):
    if not st.session_state["OPENAI_API_KEY"]:
        st.error("Please enter your OpenAI API Key to proceed.")
        st.stop()

    with st.spinner("Analysing..."):
        st.chat_message("human").write(question)
        msgs.add_user_message(question)

        try:
            # Use simplified analysis for now
            result = simple_data_analysis(question, df)
            
            if result["type"] == "table":
                response_text = result["message"]
                result_data = result["data"]
                
                # Store the dataframe
                df_index = len(st.session_state.dataframes)
                st.session_state.dataframes.append(result_data)
                msgs.add_ai_message(response_text)
                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                st.chat_message("ai").write(response_text)
                st.dataframe(result_data)
                
            else:
                response_text = result["message"]
                msgs.add_ai_message(response_text)
                st.chat_message("ai").write(response_text)
                
        except Exception as e:
            error_msg = f"An error occurred while processing your query: {str(e)}"
            st.chat_message("ai").write(error_msg)
            msgs.add_ai_message(error_msg)

# ---------------------------
# Information Section
# ---------------------------

st.sidebar.header("About This App")
st.sidebar.info("""
This is a simplified version of the Pandas Data Analyst app. 

**Current Features:**
- Basic data exploration
- Summary statistics
- Correlation analysis
- Column information

**Coming Soon:**
- Advanced data wrangling
- Interactive visualisations
- Complex data analysis

Upload your data and try these questions:
- "Show me summary statistics"
- "What are the first 10 rows?"
- "Show correlation between columns"
- "Give me information about the columns"
""") 