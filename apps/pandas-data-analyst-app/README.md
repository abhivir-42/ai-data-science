# Pandas Data Analyst AI Copilot

This Streamlit app provides an AI-powered interface for data analysis and visualisation. Upload your data and ask questions to get tables or charts as responses.

## Features

- **Data Upload**: Support for CSV and Excel files
- **Demo Data**: Built-in sample dataset for testing
- **AI-Powered Analysis**: Uses OpenAI GPT models for intelligent data analysis
- **Basic Data Operations**:
  - Summary statistics
  - Data preview and exploration
  - Correlation analysis
  - Column information
  - Interactive chat interface

## Current Status

This is a simplified version of the full Pandas Data Analyst app. The complete version with advanced data wrangling and visualisation capabilities will be available once all agents are fully integrated.

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key (you'll enter this in the app interface)

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the local Streamlit URL (usually http://localhost:8501)

3. Enter your OpenAI API key in the sidebar

4. Upload a CSV/Excel file or use the demo data

5. Ask questions about your data!

## Example Questions

- "Show me summary statistics"
- "What are the first 10 rows?"
- "Show correlation between columns"
- "Give me information about the columns"
- "Show the top 5 records"

## Coming Soon

- Advanced data wrangling capabilities
- Interactive visualisations
- Complex data transformations
- Chart generation from natural language

## Requirements

- OpenAI API key
- Python 3.8+
- See requirements.txt for full dependency list 