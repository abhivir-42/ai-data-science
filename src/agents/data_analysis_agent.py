"""
Enhanced Data Analysis Agent with Structured Outputs.

This module provides the main data analysis agent that orchestrates the complete
workflow using structured inputs/outputs, intelligent parameter mapping, and
comprehensive error handling.
"""

import asyncio
import time
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import pandas as pd
import traceback
import os
from datetime import datetime

from src.schemas import (
    DataAnalysisRequest, 
    WorkflowIntent, 
    DataAnalysisResult,
    AgentExecutionResult,
    DataQualityMetrics,
    FeatureEngineeringMetrics,
    MLModelingMetrics
)
from src.parsers import DataAnalysisIntentParser
from src.mappers import AgentParameterMapper

# Import existing agents
from src.agents.data_cleaning_agent import DataCleaningAgent
from src.agents.feature_engineering_agent import FeatureEngineeringAgent
from src.agents.ml_agents.h2o_ml_agent import H2OMLAgent

logger = logging.getLogger(__name__)


class DataAnalysisAgent:
    """
    Enhanced data analysis agent with structured outputs and intelligent orchestration.
    
    This agent replaces the current supervisor_agent.py with sophisticated workflow
    management, comprehensive parameter handling, and rich structured outputs.
    """
    
    def __init__(
        self,
        output_dir: str = "outputs",
        intent_parser_model: str = "gpt-4o-mini",
        enable_async: bool = False
    ):
        """
        Initialize the enhanced data analysis agent.
        
        Args:
            output_dir: Directory for output files
            intent_parser_model: Model to use for intent parsing
            enable_async: Whether to enable async processing
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.enable_async = enable_async
        
        # Initialize components
        self.intent_parser = DataAnalysisIntentParser(model_name=intent_parser_model)
        self.parameter_mapper = AgentParameterMapper(base_output_dir=str(self.output_dir))
        
        # Initialize agents with required models
        from langchain_openai import ChatOpenAI
        agent_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
        self.data_cleaning_agent = DataCleaningAgent(model=agent_model)
        self.feature_engineering_agent = FeatureEngineeringAgent(model=agent_model)
        self.h2o_ml_agent = H2OMLAgent(model=agent_model)
        
        # Execution tracking
        self.current_request_id = None
        self.execution_start_time = None
        
        logger.info(f"Enhanced Data Analysis Agent initialized with output_dir: {self.output_dir}")
    
    async def analyze_async(
        self,
        csv_url: str,
        user_request: str,
        **kwargs
    ) -> DataAnalysisResult:
        """
        Asynchronously perform complete data analysis workflow.
        
        Args:
            csv_url: URL to CSV file for analysis
            user_request: Natural language analysis request
            **kwargs: Additional parameters for DataAnalysisRequest
            
        Returns:
            DataAnalysisResult with comprehensive structured output
        """
        self.execution_start_time = time.time()
        
        try:
            # Step 1: Create and validate request
            request = self._create_request(csv_url, user_request, **kwargs)
            logger.info(f"Created request for: {csv_url}")
            
            # Step 2: Parse workflow intent
            intent = await self._parse_intent_async(request)
            logger.info(f"Parsed intent with confidence: {intent.intent_confidence}")
            
            # Step 3: Execute workflow
            agent_results = await self._execute_workflow_async(request, intent)
            logger.info(f"Executed {len(agent_results)} agents successfully")
            
            # Step 4: Generate comprehensive result
            result = self._generate_result(request, intent, agent_results)
            logger.info(f"Generated result with ID: {result.request_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return self._create_error_result(csv_url, user_request, str(e))
    
    def analyze(
        self,
        csv_url: str,
        user_request: str,
        **kwargs
    ) -> DataAnalysisResult:
        """
        Synchronously perform complete data analysis workflow.
        
        Args:
            csv_url: URL to CSV file for analysis
            user_request: Natural language analysis request
            **kwargs: Additional parameters for DataAnalysisRequest
            
        Returns:
            DataAnalysisResult with comprehensive structured output
        """
        if self.enable_async:
            return asyncio.run(self.analyze_async(csv_url, user_request, **kwargs))
        else:
            return self._analyze_sync(csv_url, user_request, **kwargs)
    
    def _analyze_sync(
        self,
        csv_url: str,
        user_request: str,
        **kwargs
    ) -> DataAnalysisResult:
        """Synchronous version of the analysis workflow."""
        self.execution_start_time = time.time()
        
        try:
            # Step 1: Create and validate request
            request = self._create_request(csv_url, user_request, **kwargs)
            logger.info(f"Created request for: {csv_url}")
            
            # Step 2: Parse workflow intent
            intent = self._parse_intent_sync(request)
            logger.info(f"Parsed intent with confidence: {intent.intent_confidence}")
            
            # Step 3: Execute workflow
            agent_results = self._execute_workflow_sync(request, intent)
            logger.info(f"Executed {len(agent_results)} agents successfully")
            
            # Step 4: Generate comprehensive result
            result = self._generate_result(request, intent, agent_results)
            logger.info(f"Generated result with ID: {result.request_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return self._create_error_result(csv_url, user_request, str(e))
    
    def _create_request(
        self,
        csv_url: str,
        user_request: str,
        **kwargs
    ) -> DataAnalysisRequest:
        """Create and validate a DataAnalysisRequest."""
        
        # Merge kwargs with defaults
        request_data = {
            "csv_url": csv_url,
            "user_request": user_request,
            **kwargs
        }
        
        # Create and validate request
        request = DataAnalysisRequest(**request_data)
        self.current_request_id = f"req_{self.parameter_mapper.get_timestamp()}"
        
        return request
    
    async def _parse_intent_async(self, request: DataAnalysisRequest) -> WorkflowIntent:
        """Asynchronously parse workflow intent."""
        return await self.intent_parser.parse_intent_async(
            request.user_request,
            request.csv_url
        )
    
    def _parse_intent_sync(self, request: DataAnalysisRequest) -> WorkflowIntent:
        """Synchronously parse workflow intent."""
        return self.intent_parser.parse_with_data_preview(
            request.user_request,
            request.csv_url
        )
    
    async def _execute_workflow_async(
        self,
        request: DataAnalysisRequest,
        intent: WorkflowIntent
    ) -> List[AgentExecutionResult]:
        """Asynchronously execute the complete workflow."""
        
        results = []
        current_data_path = request.csv_url
        target_variable = None
        
        # Execute data cleaning if needed
        if intent.needs_data_cleaning:
            result = await self._execute_data_cleaning_async(request, intent, current_data_path)
            results.append(result)
            if result.output_data_path:
                current_data_path = result.output_data_path
        
        # Execute feature engineering if needed
        if intent.needs_feature_engineering:
            # Determine target variable
            target_variable = (
                intent.suggested_target_variable or 
                request.target_variable or
                self._auto_detect_target_variable(current_data_path)
            )
            
            result = await self._execute_feature_engineering_async(
                request, intent, current_data_path, target_variable
            )
            results.append(result)
            if result.output_data_path:
                current_data_path = result.output_data_path
        
        # Execute ML modeling if needed
        if intent.needs_ml_modeling and target_variable:
            result = await self._execute_ml_modeling_async(
                request, intent, current_data_path, target_variable
            )
            results.append(result)
        
        return results
    
    def _execute_workflow_sync(
        self,
        request: DataAnalysisRequest,
        intent: WorkflowIntent
    ) -> List[AgentExecutionResult]:
        """Synchronously execute the complete workflow."""
        
        results = []
        current_data_path = request.csv_url
        target_variable = None
        
        # Execute data cleaning if needed
        if intent.needs_data_cleaning:
            result = self._execute_data_cleaning_sync(request, intent, current_data_path)
            results.append(result)
            if result.output_data_path:
                current_data_path = result.output_data_path
        
        # Execute feature engineering if needed
        if intent.needs_feature_engineering:
            # Determine target variable
            target_variable = (
                intent.suggested_target_variable or 
                request.target_variable or
                self._auto_detect_target_variable(current_data_path)
            )
            
            result = self._execute_feature_engineering_sync(
                request, intent, current_data_path, target_variable
            )
            results.append(result)
            if result.output_data_path:
                current_data_path = result.output_data_path
        
        # Execute ML modeling if needed
        if intent.needs_ml_modeling and target_variable:
            result = self._execute_ml_modeling_sync(
                request, intent, current_data_path, target_variable
            )
            results.append(result)
        
        return results
    
    async def _execute_data_cleaning_async(
        self,
        request: DataAnalysisRequest,
        intent: WorkflowIntent,
        data_path: str
    ) -> AgentExecutionResult:
        """Execute data cleaning agent asynchronously."""
        return self._execute_data_cleaning_sync(request, intent, data_path)
    
    def _execute_data_cleaning_sync(
        self,
        request: DataAnalysisRequest,
        intent: WorkflowIntent,
        data_path: str
    ) -> AgentExecutionResult:
        """Execute data cleaning agent synchronously."""
        
        start_time = time.time()
        
        try:
            # Load data as DataFrame
            if data_path.startswith(('http://', 'https://')):
                data_df = pd.read_csv(data_path)
            else:
                data_df = pd.read_csv(data_path)
            
            # Map parameters
            params = self.parameter_mapper.map_data_cleaning_parameters(
                request, intent, data_path
            )
            
            # Execute agent
            logger.info("Executing data cleaning agent...")
            result_dict = self.data_cleaning_agent.invoke_agent(
                data_raw=data_df,  # Pass DataFrame instead of URL
                user_instructions=params["user_instructions"],
                max_retries=3
            )
            result_str = str(result_dict)
            
            # Get the cleaned data and save it
            cleaned_df = self.data_cleaning_agent.get_data_cleaned()
            output_path = None
            
            if cleaned_df is not None:
                # Generate unique output file path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"cleaned_data_{timestamp}.csv"
                output_path = str(self.output_dir / output_filename)
                
                # Ensure output directory exists
                self.output_dir.mkdir(exist_ok=True)
                
                # Save cleaned data
                cleaned_df.to_csv(output_path, index=False)
                logger.info(f"Cleaned data saved to: {output_path}")
            else:
                logger.warning("No cleaned data was returned from the data cleaning agent")
            
            # Parse result and create metrics
            data_quality_metrics = self._extract_cleaning_metrics(result_str, data_path, output_path)
            
            execution_time = time.time() - start_time
            
            return AgentExecutionResult(
                agent_name="data_cleaning",
                execution_time_seconds=execution_time,
                success=True,
                data_quality_metrics=data_quality_metrics,
                output_data_path=output_path,
                log_messages=[result_str],
                artifacts_paths={"log": params["log_path"], "cleaned_data": output_path} if output_path else {"log": params["log_path"]}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Data cleaning failed: {e}")
            
            return AgentExecutionResult(
                agent_name="data_cleaning",
                execution_time_seconds=execution_time,
                success=False,
                error_message=str(e),
                log_messages=[traceback.format_exc()]
            )
    
    async def _execute_feature_engineering_async(
        self,
        request: DataAnalysisRequest,
        intent: WorkflowIntent,
        data_path: str,
        target_variable: Optional[str]
    ) -> AgentExecutionResult:
        """Execute feature engineering agent asynchronously."""
        return self._execute_feature_engineering_sync(request, intent, data_path, target_variable)
    
    def _execute_feature_engineering_sync(
        self,
        request: DataAnalysisRequest,
        intent: WorkflowIntent,
        data_path: str,
        target_variable: Optional[str]
    ) -> AgentExecutionResult:
        """Execute feature engineering agent synchronously."""
        
        start_time = time.time()
        
        try:
            # Load data as DataFrame
            if data_path.startswith(('http://', 'https://')):
                data_df = pd.read_csv(data_path)
            else:
                data_df = pd.read_csv(data_path)
            
            # Map parameters
            params = self.parameter_mapper.map_feature_engineering_parameters(
                request, intent, data_path, target_variable
            )
            
            # Execute agent
            logger.info("Executing feature engineering agent...")
            result_dict = self.feature_engineering_agent.invoke_agent(
                data_raw=data_df,
                user_instructions=params["user_instructions"],
                max_retries=3
            )
            result_str = str(result_dict)
            
            # Get the processed data and save it
            processed_df = self.feature_engineering_agent.get_data_engineered()
            output_path = None
            
            # Check if feature engineering actually succeeded
            if processed_df is not None and not processed_df.empty:
                # Generate unique output file path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"feature_engineered_data_{timestamp}.csv"
                output_path = str(self.output_dir / output_filename)
                
                # Ensure output directory exists
                self.output_dir.mkdir(exist_ok=True)
                
                # Save processed data
                processed_df.to_csv(output_path, index=False)
                logger.info(f"Feature engineered data saved to: {output_path}")
                
                # Parse result and create metrics
                feature_engineering_metrics = self._extract_feature_engineering_metrics(result_str, data_path, output_path)
                
                execution_time = time.time() - start_time
                
                return AgentExecutionResult(
                    agent_name="feature_engineering",
                    execution_time_seconds=execution_time,
                    success=True,
                    feature_engineering_metrics=feature_engineering_metrics,
                    output_data_path=output_path,
                    log_messages=[result_str],
                    artifacts_paths={"log": params["log_path"], "feature_engineered_data": output_path}
                )
            else:
                # Feature engineering failed - processed_df is None or empty
                logger.error("Feature engineering agent failed - no engineered data returned")
                execution_time = time.time() - start_time
                
                # Extract error message from result if available
                error_message = "Feature engineering failed after all retries"
                if "feature_engineer_error" in result_str:
                    # Try to extract the actual error
                    import re
                    # Try multiple patterns to extract error messages
                    patterns = [
                        r"'feature_engineer_error': '([^']*)'",
                        r'"feature_engineer_error": "([^"]*)"',
                        r"feature_engineer_error['\"]:\s*['\"]([^'\"]*)['\"]"
                    ]
                    
                    for pattern in patterns:
                        error_match = re.search(pattern, result_str)
                        if error_match and error_match.group(1).strip():
                            actual_error = error_match.group(1).strip()
                            error_message = f"Feature engineering failed: {actual_error}"
                            break
                
                return AgentExecutionResult(
                    agent_name="feature_engineering",
                    execution_time_seconds=execution_time,
                    success=False,
                    error_message=error_message,
                    log_messages=[result_str],
                    artifacts_paths={"log": params["log_path"]}
                )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Feature engineering failed: {e}")
            
            return AgentExecutionResult(
                agent_name="feature_engineering",
                execution_time_seconds=execution_time,
                success=False,
                error_message=str(e),
                log_messages=[traceback.format_exc()]
            )
    
    async def _execute_ml_modeling_async(
        self,
        request: DataAnalysisRequest,
        intent: WorkflowIntent,
        data_path: str,
        target_variable: str
    ) -> AgentExecutionResult:
        """Execute ML modeling agent asynchronously."""
        return self._execute_ml_modeling_sync(request, intent, data_path, target_variable)
    
    def _execute_ml_modeling_sync(
        self,
        request: DataAnalysisRequest,
        intent: WorkflowIntent,
        data_path: str,
        target_variable: str
    ) -> AgentExecutionResult:
        """Execute ML modeling agent synchronously."""
        
        start_time = time.time()
        
        try:
            # Load data as DataFrame
            if data_path.startswith(('http://', 'https://')):
                data_df = pd.read_csv(data_path)
            else:
                data_df = pd.read_csv(data_path)
            
            # Map parameters
            params = self.parameter_mapper.map_h2o_ml_parameters(
                request, intent, data_path, target_variable
            )
            
            # Execute agent
            logger.info("Executing H2O ML agent...")
            result_dict = self.h2o_ml_agent.invoke_agent(
                data_raw=data_df,
                user_instructions=params["user_instructions"],
                target_variable=target_variable,
                max_retries=3
            )
            result_str = str(result_dict)
            
            # Parse result and create metrics
            ml_metrics = self._extract_ml_metrics(result_str, params)
            
            execution_time = time.time() - start_time
            
            return AgentExecutionResult(
                agent_name="h2o_ml",
                execution_time_seconds=execution_time,
                success=True,
                ml_modeling_metrics=ml_metrics,
                model_path=params["model_directory"],
                log_messages=[result_str],
                artifacts_paths={
                    "log": params["log_path"],
                    "model_directory": params["model_directory"]
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"ML modeling failed: {e}")
            
            return AgentExecutionResult(
                agent_name="h2o_ml",
                execution_time_seconds=execution_time,
                success=False,
                error_message=str(e),
                log_messages=[traceback.format_exc()]
            )
    
    def _generate_result(
        self,
        request: DataAnalysisRequest,
        intent: WorkflowIntent,
        agent_results: List[AgentExecutionResult]
    ) -> DataAnalysisResult:
        """Generate comprehensive structured result."""
        
        total_runtime = time.time() - self.execution_start_time
        
        # Get data shape
        data_shape = self._get_data_shape(request.csv_url)
        
        # Extract metrics
        overall_quality_score = self._calculate_overall_quality_score(agent_results)
        fe_effectiveness = self._calculate_fe_effectiveness(agent_results)
        model_performance = self._calculate_model_performance(agent_results)
        
        # Generate insights and recommendations
        insights = self._generate_insights(agent_results, intent)
        recommendations = self._generate_recommendations(agent_results, intent)
        data_story = self._generate_data_story(request, intent, agent_results)
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(agent_results, intent)
        
        # Calculate analysis quality score
        analysis_quality = self._calculate_analysis_quality(agent_results, intent)
        
        # Collect warnings from multiple sources
        warnings = self._collect_warnings(agent_results)
        
        # Add warnings for invalid URLs or data access issues
        if data_shape.get("rows") == "unknown" and data_shape.get("columns") == "unknown":
            warnings.append("Could not access or load the provided dataset")
        
        # Add warnings for low confidence intent parsing
        if intent.intent_confidence < 0.3:
            warnings.append("Low confidence in understanding the request - results may not meet expectations")
        
        return DataAnalysisResult(
            total_runtime_seconds=total_runtime,
            original_request=request.user_request,
            csv_url=request.csv_url,
            data_shape=data_shape,
            workflow_intent=intent,
            agents_executed=[result.agent_name for result in agent_results],
            agent_results=agent_results,
            overall_data_quality_score=overall_quality_score,
            feature_engineering_effectiveness=fe_effectiveness,
            model_performance_score=model_performance,
            generated_files=self._collect_generated_files(agent_results),
            key_insights=insights,
            recommendations=recommendations,
            data_story=data_story,
            analysis_quality_score=analysis_quality,
            confidence_level=confidence_level,
            warnings=warnings,
            limitations=self._identify_limitations(agent_results, intent)
        )
    
    def _auto_detect_target_variable(self, data_path: str) -> Optional[str]:
        """Attempt to auto-detect target variable from data."""
        try:
            df = pd.read_csv(data_path, nrows=100)
            
            # Look for common target variable names
            target_candidates = [
                'target', 'label', 'y', 'class', 'outcome', 'result',
                'prediction', 'survived', 'price', 'value', 'score'
            ]
            
            for col in df.columns:
                if col.lower() in target_candidates:
                    return col
            
            # If no obvious target, return the last column
            return df.columns[-1]
            
        except Exception as e:
            logger.warning(f"Could not auto-detect target variable: {e}")
            return None
    
    def _find_output_file(self, filename: str) -> Optional[str]:
        """Find the output file in the output directory."""
        output_path = self.output_dir / filename
        if output_path.exists():
            return str(output_path)
        return None
    
    def _get_data_shape(self, csv_url: str) -> Dict[str, int]:
        """Get the shape of the dataset."""
        try:
            # First get column count with minimal read
            df_sample = pd.read_csv(csv_url, nrows=1)
            columns = len(df_sample.columns)
            
            # Then get full row count efficiently
            df_full = pd.read_csv(csv_url)
            rows = len(df_full)
            
            return {"rows": rows, "columns": columns}
        except Exception as e:
            logger.warning(f"Could not determine data shape for {csv_url}: {e}")
            return {"rows": "unknown", "columns": "unknown"}
    
    # Placeholder methods for metrics extraction and analysis
    # These would be implemented based on the actual agent output formats
    
    def _extract_cleaning_metrics(self, result_str: str, input_path: str, output_path: Optional[str]) -> Optional[DataQualityMetrics]:
        """Extract data quality metrics from cleaning result."""
        # This would parse the actual agent output
        return None
    
    def _extract_feature_engineering_metrics(self, result_str: str, input_path: str, output_path: Optional[str]) -> Optional[FeatureEngineeringMetrics]:
        """Extract feature engineering metrics from result."""
        # This would parse the actual agent output
        return None
    
    def _extract_ml_metrics(self, result_str: str, params: Dict[str, Any]) -> Optional[MLModelingMetrics]:
        """Extract ML modeling metrics from result."""
        # This would parse the actual agent output
        return None
    
    def _calculate_overall_quality_score(self, agent_results: List[AgentExecutionResult]) -> float:
        """Calculate overall data quality score."""
        return 0.8  # Placeholder
    
    def _calculate_fe_effectiveness(self, agent_results: List[AgentExecutionResult]) -> Optional[float]:
        """Calculate feature engineering effectiveness."""
        return 0.7  # Placeholder
    
    def _calculate_model_performance(self, agent_results: List[AgentExecutionResult]) -> Optional[float]:
        """Calculate model performance score."""
        return 0.85  # Placeholder
    
    def _generate_insights(self, agent_results: List[AgentExecutionResult], intent: WorkflowIntent) -> List[str]:
        """Generate key insights from the analysis."""
        return ["Analysis completed successfully", "Data quality is good"]
    
    def _generate_recommendations(self, agent_results: List[AgentExecutionResult], intent: WorkflowIntent) -> List[str]:
        """Generate recommendations for next steps."""
        return ["Consider additional feature engineering", "Monitor model performance"]
    
    def _generate_data_story(self, request: DataAnalysisRequest, intent: WorkflowIntent, agent_results: List[AgentExecutionResult]) -> str:
        """Generate AI narrative of the analysis."""
        return f"Successfully analyzed the dataset from {request.csv_url} according to the user's request: {request.user_request}"
    
    def _determine_confidence_level(self, agent_results: List[AgentExecutionResult], intent: WorkflowIntent) -> str:
        """Determine confidence level in results."""
        if not agent_results:
            return "low"
        success_rate = sum(1 for result in agent_results if result.success) / len(agent_results)
        if success_rate >= 0.8:
            return "high"
        elif success_rate >= 0.5:
            return "medium"
        else:
            return "low"
    
    def _calculate_analysis_quality(self, agent_results: List[AgentExecutionResult], intent: WorkflowIntent) -> float:
        """Calculate overall analysis quality score."""
        if not agent_results:
            return 0.0
        base_score = sum(1 for result in agent_results if result.success) / len(agent_results)
        confidence_bonus = intent.intent_confidence * 0.1
        return min(1.0, base_score + confidence_bonus)
    
    def _collect_generated_files(self, agent_results: List[AgentExecutionResult]) -> Dict[str, str]:
        """Collect all generated files."""
        files = {}
        for result in agent_results:
            if result.output_data_path:
                files[f"{result.agent_name}_output"] = result.output_data_path
            files.update(result.artifacts_paths)
        return files
    
    def _collect_warnings(self, agent_results: List[AgentExecutionResult]) -> List[str]:
        """Collect warnings from agent executions."""
        warnings = []
        for result in agent_results:
            warnings.extend(result.warnings)
            if not result.success:
                warnings.append(f"{result.agent_name} execution failed")
        return warnings
    
    def _identify_limitations(self, agent_results: List[AgentExecutionResult], intent: WorkflowIntent) -> List[str]:
        """Identify limitations of the analysis."""
        limitations = []
        
        if intent.intent_confidence < 0.7:
            limitations.append("Low confidence in intent parsing may affect results")
        
        failed_agents = [result.agent_name for result in agent_results if not result.success]
        if failed_agents:
            limitations.append(f"Failed to execute: {', '.join(failed_agents)}")
        
        return limitations
    
    def _create_error_result(self, csv_url: str, user_request: str, error_message: str) -> DataAnalysisResult:
        """Create an error result when analysis fails completely."""
        
        return DataAnalysisResult(
            total_runtime_seconds=time.time() - (self.execution_start_time or time.time()),
            original_request=user_request,
            csv_url=csv_url,
            data_shape={"rows": 0, "columns": 0},
            workflow_intent=WorkflowIntent(
                needs_data_cleaning=True,
                needs_feature_engineering=True,
                needs_ml_modeling=True,
                data_quality_focus=True,
                exploratory_analysis=True,
                prediction_focus=True,
                statistical_analysis=True,
                key_requirements=["Analysis failed"],
                complexity_level="moderate",
                intent_confidence=0.0
            ),
            agents_executed=[],
            agent_results=[],
            overall_data_quality_score=0.0,
            feature_engineering_effectiveness=0.0,  # Explicitly set Optional fields
            model_performance_score=0.0,  # Explicitly set Optional fields
            key_insights=["Analysis failed due to error"],
            recommendations=["Please check the data source and request"],
            data_story=f"Analysis failed: {error_message}",
            analysis_quality_score=0.0,
            confidence_level="low",
            warnings=[error_message],
            limitations=["Complete analysis failure"]
        )
    
    def analyze_from_text(self, text_input: str) -> DataAnalysisResult:
        """
        Analyze data from a single text input containing dataset info and instructions.
        
        This method intelligently parses the text to extract:
        - Dataset URLs using LLM structured outputs
        - Analysis instructions and requirements
        - Target variables and problem types
        
        Args:
            text_input: Single text containing all information
            
        Returns:
            DataAnalysisResult with comprehensive analysis
        """
        self.execution_start_time = time.time()
        
        try:
            # Step 1: Extract CSV URL using LLM structured outputs
            logger.info("Extracting dataset URL from text using LLM...")
            url_extraction = self.intent_parser.extract_dataset_url_from_text(text_input)
            
            if (url_extraction.extraction_method == "none_found" or 
                not url_extraction.extracted_csv_url or 
                url_extraction.extraction_confidence < 0.3):
                return self._create_error_result(
                    "", 
                    text_input, 
                    "Could not detect a valid CSV URL in your request. Please include a direct link to a CSV file (e.g., https://example.com/data.csv)."
                )
            
            csv_url = url_extraction.extracted_csv_url
            logger.info(f"Extracted CSV URL: {csv_url} (confidence: {url_extraction.extraction_confidence})")
            
            # Step 2: Parse workflow intent with the detected CSV URL
            logger.info("Parsing workflow intent...")
            try:
                intent = self.intent_parser.parse_with_data_preview(text_input, csv_url)
            except RuntimeError as e:
                return self._create_error_result(
                    csv_url, 
                    text_input, 
                    f"Failed to parse your request after multiple attempts. Please rephrase your request more clearly. Error: {str(e)}"
                )
            
            # Step 3: Determine adaptive runtime based on dataset size
            data_shape = self._get_data_shape(csv_url)
            adaptive_runtime = self._calculate_adaptive_runtime(data_shape)
            logger.info(f"Dataset shape: {data_shape}, Adaptive runtime: {adaptive_runtime}s")
            
            # Step 4: Create request with detected parameters and adaptive runtime
            request = DataAnalysisRequest(
                csv_url=csv_url,
                user_request=text_input,
                target_variable=intent.suggested_target_variable,
                problem_type=intent.suggested_problem_type,  # Use the intent parser's suggestion
                max_runtime_seconds=adaptive_runtime,  # Use adaptive runtime
                enable_mlflow=True,
                missing_threshold=0.4,
                outlier_detection=True,
                duplicate_removal=True,
                feature_selection=True,
                datetime_features=True,
                categorical_encoding=True
            )
            
            # Step 5: Execute the workflow based on intent flags
            logger.info(f"Executing workflow - Cleaning: {intent.needs_data_cleaning}, FE: {intent.needs_feature_engineering}, ML: {intent.needs_ml_modeling}")
            agent_results = self._execute_workflow_sync(request, intent)
            
            # Step 6: Generate comprehensive result
            result = self._generate_result(request, intent, agent_results)
            
            # Add URL extraction information to warnings if confidence is low
            if url_extraction.extraction_confidence < 0.7:
                result.warnings.append(f"Low confidence in URL extraction ({url_extraction.extraction_confidence:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return self._create_error_result("", text_input, str(e))
    
    def _calculate_adaptive_runtime(self, data_shape: Dict[str, int]) -> int:
        """
        Calculate adaptive runtime based on dataset size.
        
        Args:
            data_shape: Dictionary with 'rows' and 'columns' keys
            
        Returns:
            Adaptive runtime in seconds
        """
        try:
            rows = data_shape.get('rows', 1000)  # Default fallback
            cols = data_shape.get('columns', 10)  # Default fallback
            
            # Convert string values to int if needed
            if isinstance(rows, str):
                rows = int(rows) if rows.isdigit() else 1000
            if isinstance(cols, str):
                cols = int(cols) if cols.isdigit() else 10
            
            # Calculate adaptive runtime based on dataset size
            if rows <= 500:  # Small datasets (like flights: 144 rows)
                return 30  # 30 seconds for small datasets
            elif rows <= 5000:  # Medium datasets
                return 60  # 1 minute for medium datasets
            elif rows <= 50000:  # Large datasets
                return 120  # 2 minutes for large datasets
            else:  # Very large datasets
                return 300  # 5 minutes for very large datasets
                
        except Exception as e:
            logger.warning(f"Failed to calculate adaptive runtime: {e}, using default 60s")
            return 60  # Safe default
    