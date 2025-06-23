# Enhanced Data Analysis Agent with Structured Outputs - Implementation Plan

## 🎯 **EXECUTIVE SUMMARY**
Replace the current `supervisor_agent.py` with a sophisticated `data_analysis_agent` that uses LangChain's structured outputs for intelligent workflow orchestration. This addresses critical limitations in intent parsing, parameter utilization, and output structure.

## 🔍 **CRITICAL ANALYSIS: Current vs. Enhanced Architecture**

### **❌ Current Supervisor Agent Critical Flaws:**
After deep analysis of `supervisor_agent.py` and the individual agents:

1. **Fragile Intent Parsing (Lines 103-145)**:
   ```python
   # Current: Basic keyword matching
   cleaning_keywords = ['clean', 'preprocess', 'missing']
   needs['data_cleaning'] = any(keyword in request_lower for keyword in cleaning_keywords)
   ```
   **Problem**: Misses complex requests like "I need help with data quality issues and want to build a predictive model"

2. **Massive Parameter Under-utilization**:
   - **Data Cleaning Agent**: Only uses `user_instructions`, ignores 10+ parameters like `n_samples`, `human_in_the_loop`, `bypass_recommended_steps`
   - **Feature Engineering Agent**: Missing `target_variable` intelligence, datetime handling parameters
   - **H2O ML Agent**: Ignores MLflow integration, model directory settings, 15+ ML parameters

3. **Poor Target Variable Extraction (Lines 147-174)**:
   ```python
   # Current: Fragile regex patterns
   target_patterns = [r"target\s+(?:variable\s+)?['\"]?(\w+)['\"]?"]
   ```
   **Problem**: Fails on complex requests, doesn't understand data context

4. **Restrictive Output Structure**: Simple string concatenation doesn't capture rich agent outputs

### **✅ Enhanced Architecture Solution:**
- **LLM-powered Intent Parsing** with structured schemas
- **Complete Parameter Mapping** for all 35+ agent parameters
- **Intelligent Data Analysis** with context-aware decisions
- **Rich Structured Outputs** capturing full workflow results

## 📋 **STEP-BY-STEP IMPLEMENTATION PLAN**

### **Step 1: Schema Design & Validation Infrastructure**
**Files**: `src/schemas/data_analysis_schemas.py`

#### **1.1 Input Schema Design**
```python
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any
from enum import Enum

class DataAnalysisRequest(BaseModel):
    """Complete structured input for data analysis workflows"""
    csv_url: str = Field(description="URL to CSV file for analysis")
    user_request: str = Field(description="Natural language analysis request")
    
    # Advanced Configuration
    target_variable: Optional[str] = Field(default=None, description="Target variable for ML modeling")
    problem_type: Optional[Literal["classification", "regression", "auto"]] = Field(default="auto")
    max_runtime_seconds: Optional[int] = Field(default=300, description="Maximum runtime per agent")
    
    # Data Cleaning Preferences
    missing_threshold: Optional[float] = Field(default=0.4, description="Threshold for removing columns with missing values")
    outlier_detection: Optional[bool] = Field(default=True, description="Enable outlier detection and removal")
    
    # Feature Engineering Preferences  
    feature_selection: Optional[bool] = Field(default=True, description="Enable automatic feature selection")
    datetime_features: Optional[bool] = Field(default=True, description="Generate datetime-based features")
    
    # ML Modeling Preferences
    enable_mlflow: Optional[bool] = Field(default=True, description="Enable MLflow experiment tracking")
    model_types: Optional[List[str]] = Field(default=["GBM", "RF", "GLM"], description="H2O model types to try")
```

#### **1.2 Intent Parsing Schema**
```python
class WorkflowIntent(BaseModel):
    """LLM-parsed workflow requirements"""
    needs_data_cleaning: bool = Field(description="Requires data cleaning/preprocessing")
    needs_feature_engineering: bool = Field(description="Requires feature engineering")
    needs_ml_modeling: bool = Field(description="Requires ML model training")
    
    # Intelligent Analysis
    data_quality_focus: bool = Field(description="Primary focus on data quality issues")
    exploratory_analysis: bool = Field(description="Needs exploratory data analysis")
    prediction_focus: bool = Field(description="Primary goal is prediction/modeling")
    
    # Extracted Information
    suggested_target_variable: Optional[str] = Field(description="AI-suggested target variable")
    suggested_problem_type: Optional[str] = Field(description="AI-suggested problem type")
    key_requirements: List[str] = Field(description="Key requirements extracted from request")
```

#### **1.3 Comprehensive Output Schema**
```python
class DataAnalysisResult(BaseModel):
    """Complete structured output from data analysis workflow"""
    
    # Metadata
    request_id: str = Field(description="Unique request identifier")
    timestamp: str = Field(description="Processing timestamp")
    total_runtime_seconds: float = Field(description="Total processing time")
    
    # Input Summary
    original_request: str = Field(description="Original user request")
    csv_url: str = Field(description="Source CSV URL")
    data_shape: Dict[str, int] = Field(description="Original data dimensions")
    
    # Workflow Execution
    workflow_intent: WorkflowIntent = Field(description="Parsed workflow requirements")
    agents_executed: List[str] = Field(description="List of agents that were executed")
    
    # Agent Results
    data_cleaning_results: Optional[Dict[str, Any]] = Field(description="Data cleaning outcomes")
    feature_engineering_results: Optional[Dict[str, Any]] = Field(description="Feature engineering outcomes")
    ml_modeling_results: Optional[Dict[str, Any]] = Field(description="ML modeling outcomes")
    
    # File Outputs
    generated_files: Dict[str, str] = Field(description="Generated file paths by type")
    
    # Analysis Summary
    key_insights: List[str] = Field(description="Key insights from the analysis")
    recommendations: List[str] = Field(description="Recommendations for next steps")
    quality_score: float = Field(description="Overall analysis quality score (0-1)")
```

### **Step 2: LLM-Powered Intent Parser**
**Files**: `src/parsers/intent_parser.py`

#### **2.1 Structured Intent Parsing**
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class AdvancedIntentParser:
    def __init__(self, model: ChatOpenAI):
        self.model = model
        self.structured_model = model.with_structured_output(WorkflowIntent)
        
    def parse_user_intent(self, user_request: str, data_preview: str) -> WorkflowIntent:
        """Parse user intent using LLM with structured output"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert data scientist analyzing user requests for data analysis workflows.
            
            Based on the user's request and data preview, determine what types of analysis are needed.
            Consider the complexity and nuance of the request - don't just look for keywords.
            
            Key considerations:
            - Data quality issues require cleaning
            - Feature creation/transformation needs feature engineering  
            - Prediction/modeling goals need ML modeling
            - Complex requests may need multiple steps
            
            Provide intelligent suggestions for target variables and problem types based on context."""),
            ("human", """
            User Request: {user_request}
            
            Data Preview (first few rows and columns):
            {data_preview}
            
            Analyze this request and determine the required workflow steps.
            """)
        ])
        
        return self.structured_model.invoke(
            prompt.format_messages(
                user_request=user_request,
                data_preview=data_preview
            )
        )
```

### **Step 3: Comprehensive Parameter Mapping**
**Files**: `src/mappers/parameter_mapper.py`

#### **3.1 Agent Parameter Intelligence**
```python
class AgentParameterMapper:
    """Map user preferences to specific agent parameters"""
    
    def map_data_cleaning_params(self, request: DataAnalysisRequest, intent: WorkflowIntent) -> Dict[str, Any]:
        """Map to all 11 DataCleaningAgent parameters"""
        return {
            'model': self.model,
            'n_samples': min(50, request.max_runtime_seconds // 10),  # Adaptive sampling
            'log': True,
            'log_path': f"logs/cleaning_{request.request_id}/",
            'file_name': f"cleaned_data_{request.request_id}",
            'function_name': "clean_data",
            'overwrite': True,
            'human_in_the_loop': False,  # Automated workflow
            'bypass_recommended_steps': not intent.data_quality_focus,
            'bypass_explain_code': False,
            'checkpointer': None,  # Could add checkpointing later
            
            # Advanced parameters based on user preferences
            'missing_threshold': request.missing_threshold,
            'outlier_detection': request.outlier_detection,
        }
    
    def map_feature_engineering_params(self, request: DataAnalysisRequest, intent: WorkflowIntent) -> Dict[str, Any]:
        """Map to all FeatureEngineeringAgent parameters"""
        return {
            'model': self.model,
            'n_samples': min(50, request.max_runtime_seconds // 10),
            'log': True,
            'log_path': f"logs/features_{request.request_id}/",
            'file_name': f"engineered_data_{request.request_id}",
            'function_name': "engineer_features",
            'overwrite': True,
            'human_in_the_loop': False,
            'bypass_recommended_steps': not intent.exploratory_analysis,
            'bypass_explain_code': False,
            'checkpointer': None,
            
            # Feature engineering specific
            'target_variable': request.target_variable or intent.suggested_target_variable,
            'feature_selection': request.feature_selection,
            'datetime_features': request.datetime_features,
        }
    
    def map_h2o_ml_params(self, request: DataAnalysisRequest, intent: WorkflowIntent) -> Dict[str, Any]:
        """Map to all 15+ H2OMLAgent parameters"""
        return {
            'model': self.model,
            'n_samples': min(100, request.max_runtime_seconds // 5),
            'log': True,
            'log_path': f"logs/ml_{request.request_id}/",
            'file_name': f"ml_model_{request.request_id}",
            'function_name': "train_h2o_model",
            'overwrite': True,
            'human_in_the_loop': False,
            'bypass_recommended_steps': not intent.prediction_focus,
            'bypass_explain_code': False,
            'checkpointer': None,
            
            # H2O ML specific parameters
            'model_directory': f"models/{request.request_id}/",
            'enable_mlflow': request.enable_mlflow,
            'mlflow_tracking_uri': "http://localhost:5000",  # Default MLflow URI
            'mlflow_experiment_name': f"data_analysis_{request.request_id}",
            'target_variable': request.target_variable or intent.suggested_target_variable,
            'problem_type': request.problem_type,
            'max_runtime_secs': request.max_runtime_seconds,
            'model_types': request.model_types,
        }
```

### **Step 4: Enhanced Data Analysis Agent**
**Files**: `src/agents/data_analysis_agent.py`

#### **4.1 Core Agent Implementation**
```python
class DataAnalysisAgent:
    """Enhanced data analysis agent with structured outputs"""
    
    def __init__(self, model: ChatOpenAI = None, output_dir: str = "output/analysis/"):
        self.model = model or ChatOpenAI(model="gpt-4o-mini")
        self.output_dir = output_dir
        self.intent_parser = AdvancedIntentParser(self.model)
        self.parameter_mapper = AgentParameterMapper(self.model)
        
        # Initialize structured output model
        self.structured_model = self.model.with_structured_output(DataAnalysisResult)
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    async def analyze_data(self, request: DataAnalysisRequest) -> DataAnalysisResult:
        """Main analysis method with structured outputs"""
        start_time = time.time()
        request_id = str(uuid.uuid4())[:8]
        
        try:
            # Step 1: Download and preview data
            df, temp_path = await self._download_and_preview_data(request.csv_url)
            data_preview = self._generate_data_preview(df)
            
            # Step 2: Parse intent with LLM
            intent = self.intent_parser.parse_user_intent(request.user_request, data_preview)
            
            # Step 3: Execute workflow based on intent
            results = await self._execute_workflow(request, intent, df, request_id)
            
            # Step 4: Generate structured output
            return self._generate_structured_result(
                request, intent, results, request_id, 
                time.time() - start_time, df.shape
            )
            
        except Exception as e:
            return self._generate_error_result(request, str(e), request_id)
```

### **Step 5: Intelligent Workflow Orchestration**
**Files**: `src/orchestrators/workflow_orchestrator.py`

#### **5.1 Smart Agent Sequencing**
```python
class WorkflowOrchestrator:
    """Intelligent workflow orchestration based on intent"""
    
    async def execute_workflow(self, request: DataAnalysisRequest, intent: WorkflowIntent, 
                              df: pd.DataFrame, request_id: str) -> Dict[str, Any]:
        """Execute agents in intelligent sequence"""
        
        results = {
            'agents_executed': [],
            'data_cleaning_results': None,
            'feature_engineering_results': None,
            'ml_modeling_results': None,
            'generated_files': {},
            'errors': []
        }
        
        current_df = df.copy()
        
        # Phase 1: Data Cleaning (if needed)
        if intent.needs_data_cleaning:
            try:
                cleaning_params = self.parameter_mapper.map_data_cleaning_params(request, intent)
                current_df, cleaning_results = await self._execute_data_cleaning(
                    current_df, request.user_request, cleaning_params, request_id
                )
                results['data_cleaning_results'] = cleaning_results
                results['agents_executed'].append('data_cleaning')
            except Exception as e:
                results['errors'].append(f"Data cleaning failed: {str(e)}")
        
        # Phase 2: Feature Engineering (if needed)
        if intent.needs_feature_engineering:
            try:
                feature_params = self.parameter_mapper.map_feature_engineering_params(request, intent)
                current_df, feature_results = await self._execute_feature_engineering(
                    current_df, request.user_request, feature_params, request_id
                )
                results['feature_engineering_results'] = feature_results
                results['agents_executed'].append('feature_engineering')
            except Exception as e:
                results['errors'].append(f"Feature engineering failed: {str(e)}")
        
        # Phase 3: ML Modeling (if needed)
        if intent.needs_ml_modeling:
            target_var = request.target_variable or intent.suggested_target_variable
            if target_var and target_var in current_df.columns:
                try:
                    ml_params = self.parameter_mapper.map_h2o_ml_params(request, intent)
                    ml_results = await self._execute_ml_modeling(
                        current_df, request.user_request, ml_params, request_id
                    )
                    results['ml_modeling_results'] = ml_results
                    results['agents_executed'].append('h2o_ml')
                except Exception as e:
                    results['errors'].append(f"ML modeling failed: {str(e)}")
            else:
                results['errors'].append(f"ML modeling requested but target variable '{target_var}' not found")
        
        return results
```

### **Step 6: Advanced Output Processing**
**Files**: `src/processors/output_processor.py`

#### **6.1 Rich Result Generation**
```python
class OutputProcessor:
    """Process and enrich analysis results"""
    
    def generate_structured_result(self, request: DataAnalysisRequest, intent: WorkflowIntent,
                                  workflow_results: Dict, request_id: str, 
                                  runtime: float, data_shape: tuple) -> DataAnalysisResult:
        """Generate comprehensive structured output"""
        
        # Extract key insights using LLM
        insights = self._extract_key_insights(workflow_results, request.user_request)
        recommendations = self._generate_recommendations(workflow_results, intent)
        quality_score = self._calculate_quality_score(workflow_results)
        
        return DataAnalysisResult(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            total_runtime_seconds=runtime,
            
            original_request=request.user_request,
            csv_url=request.csv_url,
            data_shape={"rows": data_shape[0], "columns": data_shape[1]},
            
            workflow_intent=intent,
            agents_executed=workflow_results['agents_executed'],
            
            data_cleaning_results=workflow_results.get('data_cleaning_results'),
            feature_engineering_results=workflow_results.get('feature_engineering_results'),
            ml_modeling_results=workflow_results.get('ml_modeling_results'),
            
            generated_files=workflow_results.get('generated_files', {}),
            
            key_insights=insights,
            recommendations=recommendations,
            quality_score=quality_score
        )
```

### **Step 7: Testing & Validation**
**Files**: `tests/test_data_analysis_agent.py`

#### **7.1 Comprehensive Test Suite**
```python
import pytest
from src.agents.data_analysis_agent import DataAnalysisAgent
from src.schemas.data_analysis_schemas import DataAnalysisRequest

class TestDataAnalysisAgent:
    """Comprehensive test suite for enhanced data analysis agent"""
    
    @pytest.fixture
    def agent(self):
        return DataAnalysisAgent()
    
    @pytest.mark.asyncio
    async def test_simple_cleaning_request(self, agent):
        """Test basic data cleaning workflow"""
        request = DataAnalysisRequest(
            csv_url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
            user_request="Clean this dataset and handle missing values"
        )
        
        result = await agent.analyze_data(request)
        
        assert result.workflow_intent.needs_data_cleaning == True
        assert result.workflow_intent.needs_ml_modeling == False
        assert 'data_cleaning' in result.agents_executed
        assert result.quality_score > 0.5
    
    @pytest.mark.asyncio
    async def test_full_ml_pipeline(self, agent):
        """Test complete ML workflow"""
        request = DataAnalysisRequest(
            csv_url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
            user_request="Build a machine learning model to predict passenger survival",
            target_variable="Survived"
        )
        
        result = await agent.analyze_data(request)
        
        assert result.workflow_intent.needs_data_cleaning == True
        assert result.workflow_intent.needs_feature_engineering == True
        assert result.workflow_intent.needs_ml_modeling == True
        assert len(result.agents_executed) == 3
        assert result.ml_modeling_results is not None
```

## 🎯 **SUCCESS CRITERIA & VALIDATION**

### **Functional Requirements:**
- ✅ **Intent Parsing**: 95%+ accuracy on complex data analysis requests
- ✅ **Parameter Utilization**: Use all available agent parameters intelligently  
- ✅ **Structured Outputs**: Complete Pydantic validation with rich results
- ✅ **Error Handling**: Graceful failure with detailed error reporting
- ✅ **Performance**: Process typical datasets in <5 minutes

### **Technical Requirements:**
- ✅ **Schema Validation**: All inputs/outputs validated with Pydantic
- ✅ **LangChain Integration**: Use latest `with_structured_output()` methods
- ✅ **Agent Compatibility**: Work with existing agent implementations
- ✅ **Extensibility**: Easy to add new agents or parameters

## 📋 **IMPLEMENTATION CHECKLIST**

- [ ] **Step 1**: Schema design and validation infrastructure
- [ ] **Step 2**: LLM-powered intent parser with structured outputs
- [ ] **Step 3**: Comprehensive parameter mapping for all agents
- [ ] **Step 4**: Enhanced data analysis agent core implementation
- [ ] **Step 5**: Intelligent workflow orchestration
- [ ] **Step 6**: Advanced output processing with insights generation
- [ ] **Step 7**: Comprehensive testing and validation

---

**🚀 This plan addresses every limitation of the current supervisor agent and creates a production-ready, intelligent data analysis system using the latest LangChain structured outputs capabilities.** 