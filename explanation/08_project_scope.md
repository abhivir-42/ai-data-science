# AI Data Science Project Scope

## Overview
The AI Data Science project aims create a suite of AI-powered agents that collaborate to automate data science workflows. The project emphasizes ease of extension and interoperability between agents.

## Agent Ecosystem

### Current Implemented Agents
- **Data Cleaning Agent**: Analyzes datasets, recommends cleaning steps, generates and executes cleaning code, and provides cleaned data outputs.
- **Data Loader Tools Agent**: Facilitates data ingestion from various sources, handles file discovery, and prepares data for processing by other agents.

### Planned Agents
- **Data Visualization Agent**: Creates insightful visualizations based on data characteristics and user requirements.
- **Data Analysis Agent**: Performs statistical analysis and generates insights from datasets.
- **Feature Engineering Agent**: Transforms raw data into useful features for machine learning models.
- **Model Selection Agent**: Recommends appropriate ML models based on data characteristics and problem type.
- **Model Training Agent**: Trains and evaluates machine learning models with appropriate hyperparameters.
- **Model Deployment Agent**: Prepares trained models for production deployment.

## Machine Learning Integration
Machine learning is integrated at multiple levels:

1. **Foundation**: All agents are powered by large language models for reasoning and code generation.
2. **Pipeline Outputs**: The agents collectively build towards creating, training, and deploying ML models.
3. **Agent Collaboration**: The output of one agent (e.g., cleaned data) becomes the input for subsequent agents in the ML lifecycle.

## Agent Collaboration Framework
The agents work together through:

1. **Standardized Data Exchange**: Common interfaces and data formats ensure smooth handoffs between agents.
2. **Pipeline Architecture**: Agents can be chained together, with outputs from one agent feeding into inputs for another.
3. **State Management**: LangGraph provides state management across the entire processing pipeline.
4. **uAgent Integration**: Adapters allow agents to be deployed in the Fetch AI Agentverse for broader collaboration.

This ecosystem of specialized yet interoperable agents enables the complete automation of the data science workflow, from raw data ingestion to model deployment, while maintaining human-in-the-loop capabilities where desired. 