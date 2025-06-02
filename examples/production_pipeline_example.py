"""
Production Pipeline Example

This example demonstrates a production-ready data processing pipeline
using DataLoaderToolsAgent and DataCleaningAgent with:

1. Robust error handling and retry logic
2. Comprehensive logging and monitoring
3. Configuration management
4. Data validation and quality checks
5. Performance metrics and reporting
6. Batch processing capabilities

Usage:
    python examples/production_pipeline_example.py --config config.yaml
    python examples/production_pipeline_example.py --file data/sample.csv --output results/
"""

import os
import sys
import time
import logging
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.agents.data_loader_tools_agent import DataLoaderToolsAgent
from src.agents.data_cleaning_agent import DataCleaningAgent

# Load environment variables
load_dotenv()

@dataclass
class PipelineConfig:
    """Configuration for the data processing pipeline."""
    input_directory: str = "data/"
    output_directory: str = "output/"
    log_directory: str = "logs/"
    max_retries: int = 3
    batch_size: int = 10
    enable_logging: bool = True
    enable_monitoring: bool = True
    quality_threshold: float = 0.8
    timeout_seconds: int = 300
    model_name: str = "gpt-4o-mini"
    cleaning_strategy: str = "default"  # default, aggressive, conservative

@dataclass
class ProcessingResult:
    """Result of processing a single file."""
    file_path: str
    success: bool
    original_shape: tuple
    cleaned_shape: tuple
    processing_time: float
    quality_score: float
    error_message: str = ""
    cleaning_function_path: str = ""

class PipelineLogger:
    """Centralized logging for the pipeline."""
    
    def __init__(self, log_dir: str, enable_logging: bool = True):
        self.enable_logging = enable_logging
        
        if enable_logging:
            os.makedirs(log_dir, exist_ok=True)
            
            # Configure logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(os.path.join(log_dir, f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
                    logging.StreamHandler(sys.stdout)
                ]
            )
            
            self.logger = logging.getLogger('DataPipeline')
        else:
            self.logger = None
    
    def info(self, message: str):
        if self.logger:
            self.logger.info(message)
        else:
            print(f"INFO: {message}")
    
    def error(self, message: str):
        if self.logger:
            self.logger.error(message)
        else:
            print(f"ERROR: {message}")
    
    def warning(self, message: str):
        if self.logger:
            self.logger.warning(message)
        else:
            print(f"WARNING: {message}")

class DataQualityValidator:
    """Validates data quality before and after processing."""
    
    @staticmethod
    def calculate_quality_score(df: pd.DataFrame) -> float:
        """Calculate a data quality score (0-1)."""
        if df.empty:
            return 0.0
        
        # Factors for quality score
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        duplicate_ratio = df.duplicated().sum() / df.shape[0]
        
        # Quality score: higher is better
        quality_score = 1.0 - (missing_ratio * 0.5 + duplicate_ratio * 0.3)
        return max(0.0, min(1.0, quality_score))
    
    @staticmethod
    def validate_data(df: pd.DataFrame, min_rows: int = 1, min_cols: int = 1) -> tuple[bool, str]:
        """Validate basic data requirements."""
        if df.empty:
            return False, "DataFrame is empty"
        
        if df.shape[0] < min_rows:
            return False, f"Too few rows: {df.shape[0]} < {min_rows}"
        
        if df.shape[1] < min_cols:
            return False, f"Too few columns: {df.shape[1]} < {min_cols}"
        
        return True, ""

class PerformanceMonitor:
    """Monitors pipeline performance and generates reports."""
    
    def __init__(self):
        self.metrics = {
            'total_files_processed': 0,
            'successful_files': 0,
            'failed_files': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'quality_scores': [],
            'errors': []
        }
    
    def record_result(self, result: ProcessingResult):
        """Record the result of processing a file."""
        self.metrics['total_files_processed'] += 1
        
        if result.success:
            self.metrics['successful_files'] += 1
            self.metrics['quality_scores'].append(result.quality_score)
        else:
            self.metrics['failed_files'] += 1
            self.metrics['errors'].append({
                'file': result.file_path,
                'error': result.error_message
            })
        
        self.metrics['total_processing_time'] += result.processing_time
        self.metrics['average_processing_time'] = (
            self.metrics['total_processing_time'] / self.metrics['total_files_processed']
        )
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        success_rate = (
            self.metrics['successful_files'] / max(1, self.metrics['total_files_processed'])
        )
        
        avg_quality = (
            sum(self.metrics['quality_scores']) / max(1, len(self.metrics['quality_scores']))
            if self.metrics['quality_scores'] else 0.0
        )
        
        return {
            'summary': {
                'total_files': self.metrics['total_files_processed'],
                'successful_files': self.metrics['successful_files'],
                'failed_files': self.metrics['failed_files'],
                'success_rate': f"{success_rate:.2%}",
                'average_quality_score': f"{avg_quality:.3f}",
                'total_time': f"{self.metrics['total_processing_time']:.2f}s",
                'average_time_per_file': f"{self.metrics['average_processing_time']:.2f}s"
            },
            'errors': self.metrics['errors']
        }

class ProductionDataPipeline:
    """Production-ready data processing pipeline."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = PipelineLogger(config.log_directory, config.enable_logging)
        self.monitor = PerformanceMonitor() if config.enable_monitoring else None
        self.validator = DataQualityValidator()
        
        # Initialize agents
        try:
            llm = ChatOpenAI(
                model=config.model_name, 
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            self.data_loader = DataLoaderToolsAgent(model=llm)
            self.data_cleaner = DataCleaningAgent(
                model=llm,
                log=config.enable_logging,
                log_path=config.log_directory,
                human_in_the_loop=False
            )
            
            self.logger.info("Pipeline agents initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {str(e)}")
            raise
    
    def process_file(self, file_path: str, cleaning_instructions: Optional[str] = None) -> ProcessingResult:
        """Process a single file through the pipeline."""
        start_time = time.time()
        
        self.logger.info(f"Starting to process file: {file_path}")
        
        try:
            # Step 1: Load data with retries
            raw_df = self._load_data_with_retry(file_path)
            
            if raw_df is None or raw_df.empty:
                return ProcessingResult(
                    file_path=file_path,
                    success=False,
                    original_shape=(0, 0),
                    cleaned_shape=(0, 0),
                    processing_time=time.time() - start_time,
                    quality_score=0.0,
                    error_message="Failed to load data"
                )
            
            # Validate input data
            is_valid, validation_error = self.validator.validate_data(raw_df)
            if not is_valid:
                return ProcessingResult(
                    file_path=file_path,
                    success=False,
                    original_shape=raw_df.shape,
                    cleaned_shape=(0, 0),
                    processing_time=time.time() - start_time,
                    quality_score=0.0,
                    error_message=f"Data validation failed: {validation_error}"
                )
            
            # Step 2: Clean data with retries
            cleaned_df = self._clean_data_with_retry(raw_df, cleaning_instructions)
            
            if cleaned_df is None or cleaned_df.empty:
                return ProcessingResult(
                    file_path=file_path,
                    success=False,
                    original_shape=raw_df.shape,
                    cleaned_shape=(0, 0),
                    processing_time=time.time() - start_time,
                    quality_score=0.0,
                    error_message="Data cleaning failed"
                )
            
            # Step 3: Validate output data
            quality_score = self.validator.calculate_quality_score(cleaned_df)
            
            if quality_score < self.config.quality_threshold:
                self.logger.warning(f"Quality score {quality_score:.3f} below threshold {self.config.quality_threshold}")
            
            # Step 4: Save results
            output_path = self._save_results(file_path, cleaned_df)
            
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                file_path=file_path,
                success=True,
                original_shape=raw_df.shape,
                cleaned_shape=cleaned_df.shape,
                processing_time=processing_time,
                quality_score=quality_score,
                cleaning_function_path=output_path
            )
            
            self.logger.info(f"Successfully processed {file_path} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            error_msg = f"Unexpected error processing {file_path}: {str(e)}"
            self.logger.error(error_msg)
            
            return ProcessingResult(
                file_path=file_path,
                success=False,
                original_shape=(0, 0),
                cleaned_shape=(0, 0),
                processing_time=time.time() - start_time,
                quality_score=0.0,
                error_message=error_msg
            )
    
    def _load_data_with_retry(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load data with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                self.data_loader.invoke_agent(
                    user_instructions=f"Load the data file from {file_path}"
                )
                
                df = self.data_loader.get_artifacts(as_dataframe=True)
                
                if df is not None and not df.empty:
                    return df
                
            except Exception as e:
                self.logger.warning(f"Load attempt {attempt + 1} failed for {file_path}: {str(e)}")
                
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def _clean_data_with_retry(self, df: pd.DataFrame, instructions: Optional[str]) -> Optional[pd.DataFrame]:
        """Clean data with retry logic."""
        if instructions is None:
            instructions = self._get_default_cleaning_instructions()
        
        for attempt in range(self.config.max_retries):
            try:
                self.data_cleaner.invoke_agent(
                    data_raw=df,
                    user_instructions=instructions
                )
                
                cleaned_df = self.data_cleaner.get_data_cleaned()
                
                if cleaned_df is not None and not cleaned_df.empty:
                    return cleaned_df
                
            except Exception as e:
                self.logger.warning(f"Clean attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def _get_default_cleaning_instructions(self) -> str:
        """Get cleaning instructions based on strategy."""
        strategies = {
            'conservative': "Clean this data conservatively by only filling obvious missing values and removing clear duplicates",
            'aggressive': "Aggressively clean this data by removing outliers, filling all missing values, and standardizing formats",
            'default': "Apply standard data cleaning best practices for analysis"
        }
        
        return strategies.get(self.config.cleaning_strategy, strategies['default'])
    
    def _save_results(self, original_path: str, cleaned_df: pd.DataFrame) -> str:
        """Save cleaned data and metadata."""
        os.makedirs(self.config.output_directory, exist_ok=True)
        
        # Generate output filename
        file_stem = Path(original_path).stem
        output_path = os.path.join(self.config.output_directory, f"cleaned_{file_stem}.csv")
        
        # Save cleaned data
        cleaned_df.to_csv(output_path, index=False)
        
        # Save metadata
        metadata = {
            'original_file': original_path,
            'cleaned_file': output_path,
            'processing_time': datetime.now().isoformat(),
            'original_shape': cleaned_df.shape,
            'quality_score': self.validator.calculate_quality_score(cleaned_df)
        }
        
        metadata_path = os.path.join(self.config.output_directory, f"metadata_{file_stem}.yaml")
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f)
        
        return output_path
    
    def process_batch(self, file_paths: List[str], cleaning_instructions: Optional[str] = None) -> List[ProcessingResult]:
        """Process multiple files in batch."""
        results = []
        
        self.logger.info(f"Starting batch processing of {len(file_paths)} files")
        
        for i, file_path in enumerate(file_paths, 1):
            self.logger.info(f"Processing file {i}/{len(file_paths)}: {file_path}")
            
            result = self.process_file(file_path, cleaning_instructions)
            results.append(result)
            
            if self.monitor:
                self.monitor.record_result(result)
            
            # Optional: Add delay between files to prevent rate limiting
            if i < len(file_paths):
                time.sleep(1)
        
        self.logger.info("Batch processing completed")
        return results
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate a final processing report."""
        if not self.monitor:
            return {"error": "Monitoring not enabled"}
        
        report = self.monitor.generate_report()
        
        # Save report to file
        report_path = os.path.join(self.config.output_directory, f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
        with open(report_path, 'w') as f:
            yaml.dump(report, f)
        
        self.logger.info(f"Final report saved to {report_path}")
        return report

def load_config(config_path: str) -> PipelineConfig:
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return PipelineConfig(**config_data)
    else:
        return PipelineConfig()

def find_data_files(directory: str) -> List[str]:
    """Find all data files in a directory."""
    data_extensions = ['.csv', '.json', '.xlsx', '.parquet']
    files = []
    
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if any(filename.lower().endswith(ext) for ext in data_extensions):
                files.append(os.path.join(root, filename))
    
    return files

def main():
    """Main entry point for the production pipeline."""
    parser = argparse.ArgumentParser(description='Production Data Processing Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--file', help='Single file to process')
    parser.add_argument('--directory', help='Directory to process')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--instructions', help='Custom cleaning instructions')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.output:
        config.output_directory = args.output
    
    try:
        # Initialize pipeline
        pipeline = ProductionDataPipeline(config)
        
        # Determine files to process
        if args.file:
            files_to_process = [args.file]
        elif args.directory:
            files_to_process = find_data_files(args.directory)
        else:
            files_to_process = find_data_files(config.input_directory)
        
        if not files_to_process:
            print("No files found to process")
            return
        
        print(f"Found {len(files_to_process)} files to process")
        
        # Process files
        results = pipeline.process_batch(files_to_process, args.instructions)
        
        # Generate report
        if config.enable_monitoring:
            report = pipeline.generate_final_report()
            print("\n📊 Final Report:")
            print(f"Success Rate: {report['summary']['success_rate']}")
            print(f"Average Quality: {report['summary']['average_quality_score']}")
            print(f"Total Time: {report['summary']['total_time']}")
        
        # Print summary
        successful = sum(1 for r in results if r.success)
        print(f"\n✅ Processing completed: {successful}/{len(results)} files successful")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 