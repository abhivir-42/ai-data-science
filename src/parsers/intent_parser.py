"""
LLM-Powered Intent Parser for Data Analysis Workflows.

This module provides intelligent parsing of user requests using LangChain's
structured outputs to extract workflow requirements, suggest target variables,
and determine analysis complexity.
"""

import pandas as pd
import asyncio
from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.schemas import WorkflowIntent, ProblemType, DatasetExtractionRequest

logger = logging.getLogger(__name__)


class DataAnalysisIntentParser:
    """
    Intelligent parser that uses LLM with structured outputs to analyze
    user requests and extract detailed workflow requirements.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        """
        Initialize the intent parser.
        
        Args:
            model_name: OpenAI model to use for parsing
            temperature: Temperature for LLM generation (lower = more deterministic)
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize the output parser
        self.output_parser = PydanticOutputParser(pydantic_object=WorkflowIntent)
        
        # Create the prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Create the chain
        self.chain = self.prompt_template | self.llm | self.output_parser
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for intent parsing."""
        
        system_prompt = """You are an expert data scientist and workflow analyst. Your task is to analyze user requests for data analysis and extract detailed workflow requirements.

You will be given:
1. A user's natural language request for data analysis
2. Basic information about their dataset (columns, data types, shape)

Your job is to intelligently determine:
- What data analysis steps are needed (cleaning, feature engineering, ML modeling)
- The focus areas of the analysis
- Suggested target variable for ML (if applicable)
- Complexity level and confidence scores

CRITICAL PARSING RULES:
- ONLY set needs_data_cleaning=true if the user explicitly mentions cleaning, preprocessing, data quality, missing values, duplicates, or outliers
- ONLY set needs_feature_engineering=true if the user explicitly mentions features, encoding, transformations, or feature creation
- ONLY set needs_ml_modeling=true if the user explicitly mentions prediction, modeling, classification, regression, or machine learning
- Be PRECISE and LITERAL in your interpretation - don't assume additional steps unless explicitly requested
- If the user only asks for cleaning, do NOT assume they want feature engineering or ML
- If the user only asks for ML, then yes, they likely need cleaning and feature engineering as prerequisites

RESPONSE REQUIREMENTS:
- You MUST respond with valid JSON that matches the exact schema provided
- Set intent_confidence between 0.7-1.0 for clear requests, 0.3-0.6 for ambiguous ones
- Use "simple" complexity for single-step requests, "moderate" for multi-step, "complex" for advanced analysis
- Extract 1-3 key requirements as specific, actionable items
- Only suggest target variables if ML is clearly needed and you can identify a likely target from the data

EXAMPLES:
- "Clean the dataset" → needs_data_cleaning=true, needs_feature_engineering=false, needs_ml_modeling=false
- "Build a model to predict X" → needs_data_cleaning=true, needs_feature_engineering=true, needs_ml_modeling=true
- "Engineer features for the data" → needs_data_cleaning=false, needs_feature_engineering=true, needs_ml_modeling=false"""

        user_prompt = """USER REQUEST: {user_request}

DATASET INFORMATION:
- CSV URL: {csv_url}
- Dataset Shape: {data_shape}
- Column Names: {column_names}
- Data Types: {data_types}
- Sample Data: {sample_data}

Based on this information, analyze the user's request and provide a structured workflow intent analysis.

{format_instructions}"""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])
    
    async def parse_intent_async(
        self,
        user_request: str,
        csv_url: str,
        data_info: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> WorkflowIntent:
        """
        Asynchronously parse user intent from request and dataset information.
        
        Args:
            user_request: Natural language request from user
            csv_url: URL to the CSV file
            data_info: Dictionary containing dataset information
            max_retries: Maximum number of retry attempts
            
        Returns:
            WorkflowIntent object with parsed requirements
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Prepare the input data
                input_data = {
                    "user_request": user_request,
                    "csv_url": csv_url,
                    "data_shape": data_info.get("shape", "Unknown") if data_info else "Unknown",
                    "column_names": data_info.get("columns", []) if data_info else [],
                    "data_types": data_info.get("dtypes", {}) if data_info else {},
                    "sample_data": data_info.get("sample", "Not available") if data_info else "Not available",
                    "format_instructions": self.output_parser.get_format_instructions()
                }
                
                # Invoke the chain
                result = await self.chain.ainvoke(input_data)
                
                logger.info(f"Successfully parsed intent with confidence: {result.intent_confidence}")
                return result
                
            except OutputParserException as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}/{max_retries} - Failed to parse LLM output: {e}")
                if attempt < max_retries - 1:
                    continue  # Retry
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}/{max_retries} - Unexpected error in intent parsing: {e}")
                if attempt < max_retries - 1:
                    continue  # Retry
        
        # If all retries failed, raise the last error instead of using fallback
        logger.error(f"Intent parsing failed after {max_retries} attempts. Last error: {last_error}")
        raise RuntimeError(f"Intent parsing failed after {max_retries} attempts: {last_error}")
    
    def parse_intent(
        self,
        user_request: str,
        csv_url: str,
        data_info: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> WorkflowIntent:
        """
        Synchronously parse user intent from request and dataset information.
        
        Args:
            user_request: Natural language request from user
            csv_url: URL to the CSV file
            data_info: Dictionary containing dataset information
            max_retries: Maximum number of retry attempts
            
        Returns:
            WorkflowIntent object with parsed requirements
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Prepare the input data
                input_data = {
                    "user_request": user_request,
                    "csv_url": csv_url,
                    "data_shape": data_info.get("shape", "Unknown") if data_info else "Unknown",
                    "column_names": data_info.get("columns", []) if data_info else [],
                    "data_types": data_info.get("dtypes", {}) if data_info else {},
                    "sample_data": data_info.get("sample", "Not available") if data_info else "Not available",
                    "format_instructions": self.output_parser.get_format_instructions()
                }
                
                # Use synchronous invoke
                result = self.chain.invoke(input_data)
                
                logger.info(f"Successfully parsed intent with confidence: {result.intent_confidence}")
                return result
                
            except OutputParserException as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}/{max_retries} - Failed to parse LLM output: {e}")
                if attempt < max_retries - 1:
                    continue  # Retry
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}/{max_retries} - Unexpected error in intent parsing: {e}")
                if attempt < max_retries - 1:
                    continue  # Retry
        
        # If all retries failed, raise the last error instead of using fallback
        logger.error(f"Intent parsing failed after {max_retries} attempts. Last error: {last_error}")
        raise RuntimeError(f"Intent parsing failed after {max_retries} attempts: {last_error}")
    

    
    def get_data_preview(self, csv_url: str, max_rows: int = 5) -> Dict[str, Any]:
        """
        Get a preview of the dataset for better intent parsing.
        
        Args:
            csv_url: URL to the CSV file
            max_rows: Maximum number of rows to sample
            
        Returns:
            Dictionary with dataset information
        """
        try:
            # Read the dataset
            df = pd.read_csv(csv_url, nrows=max_rows * 2)  # Read a bit more for sampling
            
            # Get basic info
            data_info = {
                "shape": {"rows": len(df), "columns": len(df.columns)},
                "columns": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "sample": df.head(max_rows).to_dict('records')
            }
            
            # Add missing value info
            missing_info = df.isnull().sum()
            data_info["missing_values"] = {
                col: int(count) for col, count in missing_info.items() if count > 0
            }
            
            return data_info
            
        except Exception as e:
            logger.error(f"Failed to get data preview: {e}")
            return {
                "shape": "Unknown",
                "columns": [],
                "dtypes": {},
                "sample": "Could not load data preview",
                "missing_values": {}
            }
    
    def parse_with_data_preview(
        self,
        user_request: str,
        csv_url: str,
        max_preview_rows: int = 5
    ) -> WorkflowIntent:
        """
        Parse intent with automatic data preview for better analysis.
        
        Args:
            user_request: Natural language request from user
            csv_url: URL to the CSV file
            max_preview_rows: Maximum rows to preview
            
        Returns:
            WorkflowIntent object with parsed requirements
        """
        # Get data preview
        data_info = self.get_data_preview(csv_url, max_preview_rows)
        
        # Parse intent with the data information
        return self.parse_intent(user_request, csv_url, data_info)

    def extract_dataset_url_from_text(self, text_input: str) -> DatasetExtractionRequest:
        """
        Extract dataset URL from text using LLM with structured outputs.
        
        Args:
            text_input: User's text input that may contain dataset information
            
        Returns:
            DatasetExtractionRequest with extracted URL and metadata
        """
        try:
            # Create extraction parser
            extraction_parser = PydanticOutputParser(pydantic_object=DatasetExtractionRequest)
            
            # Create extraction prompt
            extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a dataset URL extraction expert. Your task is to analyze user text and extract CSV dataset URLs.

EXTRACTION RULES:
1. Look for explicit HTTP/HTTPS URLs that end with .csv
2. Only extract URLs that clearly point to CSV files
3. Do NOT invent or generate URLs that aren't explicitly mentioned
4. If no URL is found, set extraction_method to "none_found" and provide an empty string for extracted_csv_url
5. Be conservative - only extract URLs you are confident about

IMPORTANT: Do not create or infer URLs from dataset names. Only extract explicit URLs that are mentioned in the text."""),
                ("user", """TEXT TO ANALYZE: {text_input}

Extract the CSV dataset URL from this text. If no explicit CSV URL is found, indicate that none was found.

{format_instructions}""")
            ])
            
            # Create extraction chain
            extraction_chain = extraction_prompt | self.llm | extraction_parser
            
            # Extract URL
            result = extraction_chain.invoke({
                "text_input": text_input,
                "format_instructions": extraction_parser.get_format_instructions()
            })
            
            logger.info(f"URL extraction result: {result.extraction_method} with confidence {result.extraction_confidence}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to extract dataset URL: {e}")
            # Return fallback result
            return DatasetExtractionRequest(
                extracted_csv_url="",
                extraction_confidence=0.0,
                extraction_method="none_found",
                extraction_notes=f"Extraction failed: {str(e)}"
            )


# Convenience function for quick usage
def parse_data_analysis_intent(
    user_request: str,
    csv_url: str,
    data_info: Optional[Dict[str, Any]] = None,
    model_name: str = "gpt-4o-mini"
) -> WorkflowIntent:
    """
    Quick function to parse data analysis intent.
    
    Args:
        user_request: Natural language request from user
        csv_url: URL to the CSV file
        data_info: Optional dataset information
        model_name: OpenAI model to use
        
    Returns:
        WorkflowIntent object with parsed requirements
    """
    parser = DataAnalysisIntentParser(model_name=model_name)
    
    if data_info is None:
        return parser.parse_with_data_preview(user_request, csv_url)
    else:
        return parser.parse_intent(user_request, csv_url, data_info) 