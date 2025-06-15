# Exploratory Data Analysis (EDA) Copilot App

This Streamlit app provides an AI-powered interface for comprehensive exploratory data analysis. Upload your data and ask questions to get insights, visualisations, and reports.

## Features

- **Data Upload**: Support for CSV and Excel files
- **Demo Data**: Built-in sample dataset for testing
- **AI-Powered Analysis**: Uses OpenAI GPT models for intelligent data exploration
- **Comprehensive EDA Tools**:
  - Data summaries and descriptions
  - Missing data visualisations
  - Correlation analysis
  - Interactive reports (Sweetviz, D-Tale)
  - Statistical summaries

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

5. Ask questions about your data in natural language!

## Example Questions

- "What tools do you have access to?"
- "Explain the dataset"
- "Describe the dataset"
- "Analyze missing data in the dataset"
- "Generate a correlation funnel using [target_column] as the target"
- "Generate a Sweetviz report"
- "Generate a D-Tale report"

## Requirements

- OpenAI API key
- Python 3.8+
- See requirements.txt for full dependency list 