# ML Prediction Enhancement Implementation Plan

**Objective**: Extend the enhanced uAgent to support ML model predictions and model-based queries within the same chat session, leveraging the existing session management pattern.

## 🎯 Current State Analysis

### Existing Session Management Pattern
The enhanced uAgent already has an excellent session management system:

```python
# Current session management in EnhancedDataAnalysisUAgent
self._last_cleaned_data = None              # Stores cleaned DataFrame
self._last_processed_timestamp = None       # Tracks processing time
```

### Existing Workflow
1. User requests analysis → DataAnalysisAgent executes workflow → Results formatted
2. Cleaned data stored in session for follow-up requests
3. Session expires after configurable timeout (default: 1 hour)

## 🚀 Enhancement Strategy

### Extension Approach: Follow Existing Pattern
Extend the session management to also store ML model information alongside cleaned data:

```python
# Extended session management (NEW)
self._last_trained_model = None             # Stores trained model metadata
self._last_model_timestamp = None           # Tracks model training time
```

## 📋 Implementation Plan

### Phase 1: Extend Schema & Session Management

#### 1.1 Use Existing MLModelingMetrics (No New Schema Needed!)
**File**: `ai-data-science/src/schemas/data_analysis_schemas.py` - **NO CHANGES NEEDED**

✅ **IMPORTANT DISCOVERY**: The existing `MLModelingMetrics` already contains all the rich H2O data we need:
- `model_path`, `leaderboard`, `top_model_metrics`, `total_models_trained`
- `generated_code`, `recommended_steps`, `workflow_summary`, `model_architecture`
- `enhanced_feature_importance` and all other comprehensive H2O AutoML data

**We can reuse the existing schema completely!** This simplifies the implementation significantly.

#### 1.2 Extend Enhanced uAgent Session Management
**File**: `ai-data-science/src/uagent_v2/enhanced_uagent.py`

```python
class EnhancedDataAnalysisUAgent:
    def __init__(self, config: Optional[UAgentConfig] = None):
        # ... existing initialization ...
        
        # EXTENDED: Session management for ML models
        self._last_cleaned_data = None
        self._last_processed_timestamp = None
        
        # NEW: ML model session management  
        self._last_trained_model = None          # MLModelingMetrics object from successful ML training
        self._last_model_timestamp = None        # When model was trained
        self._last_training_result = None        # AgentExecutionResult with ML metrics
        self._last_target_variable = None        # Target variable used for training
        
        # NEW: Initialize prediction formatter (ADD TO EXISTING __init__)
        from .prediction_formatters import PredictionResponseFormatter
        self.prediction_formatter = PredictionResponseFormatter(self.config)
        
    def _store_ml_model_if_available(self, result: DataAnalysisResult):
        """Store trained ML model information for follow-up predictions."""
        try:
            # Find ML agent result
            ml_agent_result = None
            for agent_result in result.agent_results:
                if agent_result.agent_name == "h2o_ml" and agent_result.success:
                    ml_agent_result = agent_result
                    break
            
            if ml_agent_result and ml_agent_result.ml_modeling_metrics:
                metrics = ml_agent_result.ml_modeling_metrics
                
                # Store the existing MLModelingMetrics directly (no new schema needed!)
                self._last_trained_model = metrics
                self._last_model_timestamp = time.time()
                self._last_training_result = ml_agent_result
                self._last_target_variable = self._extract_target_variable(result)
                
                self.logger.info(f"Stored trained model session: {metrics.best_model_id}")
                self.logger.info(f"Model path: {metrics.model_path}")
                self.logger.info(f"Target variable: {self._last_target_variable}")
                
        except Exception as e:
            self.logger.warning(f"Could not store ML model session: {e}")
    
    def _extract_target_variable(self, result: DataAnalysisResult) -> Optional[str]:
        """Extract target variable from the analysis result."""
        # Try workflow intent first
        if result.workflow_intent.suggested_target_variable:
            return result.workflow_intent.suggested_target_variable
        
        # Try to find from ML agent execution logs
        for agent_result in result.agent_results:
            if agent_result.agent_name == "h2o_ml" and agent_result.success:
                # Target variable might be in log messages or metadata
                # This would need to be extracted from the actual agent execution
                pass
        
        return None
    
    def _is_model_session_expired(self) -> bool:
        """Check if the ML model session has expired."""
        if not self._last_model_timestamp:
            return True
        
        session_age = time.time() - self._last_model_timestamp
        max_age = self.config.session_timeout_hours * 3600
        return session_age > max_age
    
    def _has_trained_model(self) -> bool:
        """Check if we have a valid trained model in session."""
        return (self._last_trained_model is not None and 
                not self._is_model_session_expired() and
                self._last_trained_model.model_path is not None)
```

### Phase 2: Enhance Intent Parser for Prediction Recognition

#### 2.1 Extend WorkflowIntent Schema
**File**: `ai-data-science/src/schemas/data_analysis_schemas.py`

```python
class WorkflowIntent(BaseModel):
    # ... existing fields ...
    
    # NEW: Prediction-specific intent fields
    needs_prediction: bool = Field(
        default=False,
        description="Request requires making predictions with existing model"
    )
    needs_model_analysis: bool = Field(
        default=False,
        description="Request requires analyzing existing model results/insights"
    )
    
    # NEW: Prediction details
    prediction_data_source: Optional[str] = Field(
        default=None,
        description="Data source for prediction (CSV URL, inline data, etc.)"
    )
    #AB
    prediction_type: Optional[Literal["single_prediction", "batch_prediction", "model_analysis"]] = Field(
        default=None,
        description="Type of prediction request"
    )
    extracted_prediction_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extracted prediction input data from user request"
    )
```

#### 2.2 Enhance Intent Parser
**File**: `ai-data-science/src/parsers/intent_parser.py`

```python
def _create_prompt_template(self) -> ChatPromptTemplate:
    system_prompt = """You are an expert data scientist and workflow analyst. Your task is to analyze user requests for data analysis and extract detailed workflow requirements.

CRITICAL PARSING RULES:
- ONLY set needs_data_cleaning=true if the user explicitly mentions cleaning, preprocessing, data quality, missing values, duplicates, or outliers
- ONLY set needs_feature_engineering=true if the user explicitly mentions features, encoding, transformations, or feature creation
- ONLY set needs_ml_modeling=true if the user explicitly mentions prediction, modeling, classification, regression, or machine learning
- NEW: Set needs_prediction=true if user asks for predictions, forecasts, or wants to use a model for new data
- NEW: Set needs_model_analysis=true if user asks questions about model performance, feature importance, or model insights

PREDICTION REQUEST PATTERNS:
- "predict", "forecast", "classify", "estimate" with new data
- "what would happen if", "predict for", "classify this"
- "use the model to", "make prediction", "run inference"
- Questions about model: "what features are important", "why did the model predict", "model performance"

PREDICTION DATA EXTRACTION:
- Look for CSV URLs for batch prediction and set prediction_data_source
- Look for inline data like: age=25, income=50000 and set extracted_prediction_data
- Look for "predict for" followed by data values and parse into key-value pairs
- Set prediction_type to "single_prediction", "batch_prediction", or "model_analysis"
- Extract structured prediction input data into extracted_prediction_data field
"""
```

### Phase 3: Create ML Prediction Engine

#### 3.1 Create ML Prediction Agent
**File**: `ai-data-science/src/agents/ml_prediction_agent.py`

```python
class MLPredictionAgent:
    """Agent for making predictions with trained H2O models."""
    
    def __init__(self, model_metrics: MLModelingMetrics, target_variable: str, config: UAgentConfig):
        self.model_metrics = model_metrics
        self.target_variable = target_variable
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._h2o_model = None
        
    def load_model(self):
        """Load the H2O model for predictions."""
        try:
            import h2o
            h2o.init()
            
            self._h2o_model = h2o.load_model(self.model_metrics.model_path)
            self.logger.info(f"Loaded H2O model: {self.model_metrics.best_model_id}")
            
        except Exception as e:
            raise MLPredictionError(f"Failed to load model: {e}")
    
    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction for a single data point."""
        try:
            if self._h2o_model is None:
                self.load_model()
            
            # Convert input to H2O frame
            df = pd.DataFrame([input_data])
            h2o_frame = h2o.H2OFrame(df)
            
            # Make prediction
            predictions = self._h2o_model.predict(h2o_frame)
            
            # Convert to results
            pred_df = predictions.as_data_frame()
            
            # Format results based on problem type  
            # Determine problem type from model metrics or target variable characteristics
            problem_type = self._determine_problem_type()
            if problem_type == "classification":
                return self._format_classification_result(pred_df, input_data)
            else:
                return self._format_regression_result(pred_df, input_data)
                
        except Exception as e:
            raise MLPredictionError(f"Prediction failed: {e}")
    
    def predict_batch(self, data_source: str) -> Dict[str, Any]:
        """Make predictions for batch data from CSV URL."""
        try:
            if self._h2o_model is None:
                self.load_model()
            
            # Load data
            df = pd.read_csv(data_source)
            h2o_frame = h2o.H2OFrame(df)
            
            # Make predictions
            predictions = self._h2o_model.predict(h2o_frame)
            pred_df = predictions.as_data_frame()
            
            # Combine with original data
            result_df = pd.concat([df, pred_df], axis=1)
            
            # Save results
            output_path = self._save_prediction_results(result_df)
            
            return {
                "prediction_type": "batch",
                "input_rows": len(df),
                "output_path": output_path,
                "predictions_summary": self._summarize_batch_predictions(pred_df),
                "download_link": output_path
            }
            
        except Exception as e:
            raise MLPredictionError(f"Batch prediction failed: {e}")
    
    def analyze_model(self, query: str) -> Dict[str, Any]:
        """Answer questions about the trained model."""
        try:
            # Use LLM to answer model-related questions
            from langchain_openai import ChatOpenAI
            
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
            
            model_info = {
                "architecture": self.model_metrics.model_architecture,
                "features": self.model_metrics.features_used,
                "performance": self.model_metrics.best_model_score,
                "target": self.target_variable,
                "training_duration": self.model_metrics.training_time_seconds,
                # Rich H2O data for comprehensive analysis
                "leaderboard": self.model_metrics.leaderboard,
                "top_model_metrics": self.model_metrics.top_model_metrics,
                "total_models_trained": self.model_metrics.total_models_trained,
                "feature_importance": self.model_metrics.enhanced_feature_importance,
                "generated_code": self.model_metrics.generated_code,
                "recommended_steps": self.model_metrics.recommended_steps,
                "workflow_summary": self.model_metrics.workflow_summary
            }
            
            prompt = f"""
            Answer the user's question about this trained ML model using the comprehensive H2O AutoML data:
            
            Model Information:
            - Architecture: {model_info['architecture']}
            - Target Variable: {model_info['target']}
            - Features Used: {', '.join(model_info['features'])}
            - Performance Score: {model_info['performance']}
            - Training Duration: {model_info['training_duration']} seconds
            - Total Models Trained: {model_info['total_models_trained']}
            
            Feature Importance: {model_info['feature_importance']}
            Top Model Metrics: {model_info['top_model_metrics']}
            Leaderboard: {model_info['leaderboard']}
            Workflow Summary: {model_info['workflow_summary']}
            
            User Question: {query}
            
            Provide a helpful, accurate answer based on the comprehensive model information above.
            Use the feature importance, leaderboard, and metrics to give detailed insights.
            """
            
            response = llm.invoke(prompt)
            
            return {
                "analysis_type": "model_question",
                "question": query,
                "answer": response.content,
                "model_info": model_info
            }
            
        except Exception as e:
            raise MLPredictionError(f"Model analysis failed: {e}")
    
    def _determine_problem_type(self) -> str:
        """Determine if this is a classification or regression problem."""
        # Try to determine from model architecture
        if self.model_metrics.model_architecture:
            arch = self.model_metrics.model_architecture.lower()
            if "classification" in arch:
                return "classification"
            elif "regression" in arch:
                return "regression"
        
        # Try to determine from metrics
        if self.model_metrics.top_model_metrics:
            metrics = self.model_metrics.top_model_metrics
            # AUC suggests classification, RMSE suggests regression
            if 'auc' in str(metrics).lower() or 'logloss' in str(metrics).lower():
                return "classification"
            elif 'rmse' in str(metrics).lower() or 'mae' in str(metrics).lower():
                return "regression"
        
        # Default fallback
        return "classification"
    
    def _format_classification_result(self, pred_df: pd.DataFrame, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format classification prediction results."""
        # Get prediction and probability
        prediction = pred_df.iloc[0]['predict'] if 'predict' in pred_df.columns else pred_df.iloc[0, 0]
        
        # Get probability if available
        probability = None
        prob_cols = [col for col in pred_df.columns if col.startswith('p') and col != 'predict']
        if prob_cols:
            probability = pred_df.iloc[0][prob_cols[0]]
        
        return {
            "prediction_type": "single_prediction", 
            "target_variable": self.target_variable,
            "prediction": prediction,
            "probability": probability,
            "input_data": input_data,
            "model_architecture": self.model_metrics.model_architecture,
            "model_score": self.model_metrics.best_model_score
        }
    
    def _format_regression_result(self, pred_df: pd.DataFrame, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format regression prediction results."""
        prediction = pred_df.iloc[0]['predict'] if 'predict' in pred_df.columns else pred_df.iloc[0, 0]
        
        return {
            "prediction_type": "single_prediction",
            "target_variable": self.target_variable,
            "prediction": prediction,
            "input_data": input_data,
            "model_architecture": self.model_metrics.model_architecture,
            "model_score": self.model_metrics.best_model_score
        }
    
    def _save_prediction_results(self, result_df: pd.DataFrame) -> str:
        """Save batch prediction results to file."""
        import os
        from datetime import datetime
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_predictions_{timestamp}.csv"
        
        # Use config output directory or create default
        output_dir = getattr(self.config, 'output_dir', 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, filename)
        result_df.to_csv(output_path, index=False)
        
        return output_path
    
    def _summarize_batch_predictions(self, pred_df: pd.DataFrame) -> Dict[str, Any]:
        """Create summary statistics for batch predictions."""
        summary = {}
        
        if 'predict' in pred_df.columns:
            predictions = pred_df['predict']
            
            # For classification
            if predictions.dtype == 'object' or predictions.nunique() < 20:
                summary['prediction_counts'] = predictions.value_counts().to_dict()
            
            # For regression
            else:
                summary['prediction_stats'] = {
                    'mean': float(predictions.mean()),
                    'std': float(predictions.std()),
                    'min': float(predictions.min()),
                    'max': float(predictions.max())
                }
        
        return summary


# Custom exception for ML prediction errors
class MLPredictionError(Exception):
    """Custom exception for ML prediction failures."""
    pass
```

#### 3.2 Create Prediction Response Formatters
**File**: `ai-data-science/src/uagent_v2/prediction_formatters.py`

```python
class PredictionResponseFormatter:
    """Format prediction results for user-friendly display."""
    
    def __init__(self, config: UAgentConfig):
        self.config = config
    
    def format_single_prediction(self, prediction_result: Dict[str, Any]) -> str:
        """Format single prediction result."""
        lines = [
            "🔮 **PREDICTION RESULT**",
            "=" * 40,
            "",
            f"📊 **Input Data**:",
        ]
        
        # Display input features
        for feature, value in prediction_result["input_data"].items():
            lines.append(f"   • {feature}: {value}")
        
        lines.extend([
            "",
            f"🎯 **Prediction**:",
            f"   • **{prediction_result['target_variable']}**: {prediction_result['prediction']}",
        ])
        
        # Add confidence/probability if classification
        if prediction_result.get("probability"):
            lines.append(f"   • **Confidence**: {prediction_result['probability']:.2%}")
        
        lines.extend([
            "",
            f"🤖 **Model Used**: {prediction_result['model_architecture']}",
            f"📈 **Model Performance**: {prediction_result['model_score']:.3f}",
            ""
        ])
        
        return "\n".join(lines)
    
    def format_batch_prediction(self, prediction_result: Dict[str, Any]) -> str:
        """Format batch prediction results."""
        lines = [
            "🔮 **BATCH PREDICTION COMPLETE**",
            "=" * 50,
            "",
            f"📊 **Results Summary**:",
            f"   • **Rows Processed**: {prediction_result['input_rows']:,}",
            f"   • **Predictions Made**: {prediction_result['input_rows']:,}",
            "",
            f"📁 **Download Results**:",
            f"   • **File**: {prediction_result['output_path']}",
            "",
        ]
        
        # Add prediction summary
        if prediction_result.get("predictions_summary"):
            lines.extend([
                "📈 **Prediction Summary**:",
            ])
            for key, value in prediction_result["predictions_summary"].items():
                lines.append(f"   • {key}: {value}")
        
        return "\n".join(lines)
    
    def format_model_analysis(self, analysis_result: Dict[str, Any]) -> str:
        """Format model analysis response."""
        lines = [
            "🧠 **MODEL ANALYSIS**",
            "=" * 40,
            "",
            f"❓ **Question**: {analysis_result['question']}",
            "",
            f"💡 **Answer**:",
            f"{analysis_result['answer']}",
            "",
            f"📊 **Model Details**:",
        ]
        
        model_info = analysis_result["model_info"]
        lines.extend([
            f"   • **Architecture**: {model_info['architecture']}",
            f"   • **Target**: {model_info['target']}",
            f"   • **Performance**: {model_info['performance']:.3f}",
            f"   • **Features**: {len(model_info['features'])} features used",
            ""
        ])
        
        return "\n".join(lines)
```

### Phase 4: Integrate with Enhanced uAgent

#### 4.1 Extend Query Processing
**File**: `ai-data-science/src/uagent_v2/enhanced_uagent.py`

```python
def process_query(self, query: Union[str, Dict[str, Any]]) -> str:
    """Process a user query with ML prediction support."""
    try:
        # ... existing code ...
        
        query_lower = query_text.lower()
        
        # NEW: Use LLM intent parser to determine query type
        intent = self.intent_parser.parse_with_data_preview(query_text, "")
        
        # Handle ML prediction requests
        if intent.needs_prediction:
            return self._handle_prediction_request(query_text, intent)
        
        # Handle model analysis questions
        if intent.needs_model_analysis:
            return self._handle_model_analysis_request(query_text, intent)
        
        # COMMENTED OUT: Data delivery handling (not needed for ML prediction)
        # if any(phrase in query_lower for phrase in [
        #     'send my data', 'provide my cleaned data', 'show me my processed data',
        #     'my cleaned dataset', 'give me my data', 'deliver my data',
        #     'send rows', 'send columns', 'data in chunks', 'split my data'
        # ]):
        #     return self._handle_data_delivery_request(query_text)
        
        # Process the main analysis request
        return self._process_analysis_request(query_text)
        
    except Exception as e:
        self.logger.error(f"Query processing failed: {e}", exc_info=True)
        return self._create_error_response(e)

def _handle_prediction_request(self, query: str, intent: WorkflowIntent) -> str:
    """Handle prediction requests using trained model."""
    try:
        # Check if we have a trained model
        if not self._has_trained_model():
            return self._create_no_model_response()
        
        # Intent already parsed - use it directly
        # Create prediction agent
        from src.agents.ml_prediction_agent import MLPredictionAgent
        prediction_agent = MLPredictionAgent(
            self._last_trained_model, 
            self._last_target_variable, 
            self.config
        )
        
        # Execute prediction based on intent
        if intent.prediction_type == "single_prediction":
            result = prediction_agent.predict_single(intent.extracted_prediction_data)
            return self.prediction_formatter.format_single_prediction(result)
            
        elif intent.prediction_type == "batch_prediction":
            result = prediction_agent.predict_batch(intent.prediction_data_source)
            return self.prediction_formatter.format_batch_prediction(result)
            
        else:
            return "🚫 Could not understand the prediction request."
            
    except Exception as e:
        self.logger.error(f"Prediction request failed: {e}")
        return self._create_prediction_error_response(e)

def _handle_model_analysis_request(self, query: str, intent: WorkflowIntent) -> str:
    """Handle model analysis questions."""
    try:
        # Check if we have a trained model
        if not self._has_trained_model():
            return self._create_no_model_response()
        
        # Create prediction agent for analysis
        from src.agents.ml_prediction_agent import MLPredictionAgent
        prediction_agent = MLPredictionAgent(
            self._last_trained_model, 
            self._last_target_variable, 
            self.config
        )
        
        # Analyze model (intent already parsed)
        result = prediction_agent.analyze_model(query)
        return self.prediction_formatter.format_model_analysis(result)
        
    except Exception as e:
        self.logger.error(f"Model analysis failed: {e}")
        return self._create_prediction_error_response(e)

def _create_no_model_response(self) -> str:
    """Response when no trained model is available."""
    return """
🚫 **No Trained Model Found**

I don't have a trained ML model in this session to make predictions. 

**To get started:**
1. First train a model: "Clean and build ML model using https://example.com/data.csv to predict target_column"
2. Then make predictions: "Predict target_column for age=25, income=50000"

**Example workflow:**
```
You: "Train a model using https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv to predict Survived"
Me: [Trains model and shows results]
You: "Predict survival for Age=25, Sex=male, Pclass=3"
Me: [Uses trained model to make prediction]
```
"""

def _create_prediction_error_response(self, error: Exception) -> str:
    """Create error response for prediction failures."""
    return f"""
🚫 **Prediction Error**

Sorry, I encountered an issue while making the prediction: {str(error)}

**Common solutions:**
1. Check that your input data format matches the training data
2. Ensure all required features are provided
3. Try retraining the model if data format has changed

**Need help?** Try asking: "What features does the model expect?" or retrain with: "Train a new model using [your dataset URL]"
"""
```

#### 4.2 Update Main Process to Store Model
**File**: `ai-data-science/src/uagent_v2/enhanced_uagent.py`

```python
def _process_analysis_request(self, query: str) -> str:
    """Process the main data analysis request following the original pattern."""
    try:
        # Direct invocation of the underlying DataAnalysisAgent
        result = self.data_analysis_agent.analyze_from_text(query)
        
        # Store cleaned data for potential follow-up requests (existing)
        self._store_cleaned_data_if_available()
        
        # NEW: Store ML model for potential predictions
        self._store_ml_model_if_available(result)
        
        # Format the structured result for uAgent compatibility
        return self.result_formatter.format_analysis_result_enhanced(result)
        
    except Exception as e:
        self.logger.error(f"Analysis request failed: {e}", exc_info=True)
        return self._create_error_response(e)
```

### Phase 5: Testing & Validation

#### 5.1 Create Test Suite
**File**: `ai-data-science/test_ml_prediction_functionality.py`

```python
def test_ml_prediction_workflow():
    """Test complete ML prediction workflow."""
    
    # Test 1: Train model
    enhanced_uagent = EnhancedDataAnalysisUAgent()
    
    # Train model
    train_result = enhanced_uagent.process_query(
        "Train model using https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv to predict Survived"
    )
    assert "ML MODEL TRAINING COMPLETE" in train_result
    assert enhanced_uagent._has_trained_model()
    
    # Test 2: Single prediction
    pred_result = enhanced_uagent.process_query(
        "Predict Survived for Age=25, Sex=male, Pclass=3, Fare=7.25"
    )
    assert "PREDICTION RESULT" in pred_result
    
    # Test 3: Model analysis
    analysis_result = enhanced_uagent.process_query(
        "What are the most important features for survival prediction?"
    )
    assert "MODEL ANALYSIS" in analysis_result
    
    # Test 4: Session expiration
    enhanced_uagent._last_model_timestamp = time.time() - (2 * 3600)  # 2 hours ago
    expired_result = enhanced_uagent.process_query("Predict survival for Age=30")
    assert "No Trained Model Found" in expired_result
```

## 🎯 Implementation Timeline

### Week 1: Schema & Session Management
- [ ] Extend `WorkflowIntent` for predictions (2 new fields)
- [ ] Update enhanced uAgent session management (4 new instance variables)  
- [ ] Add model storage methods (1 new method, reuse existing MLModelingMetrics)
- [ ] Add prediction formatter initialization to enhanced uAgent __init__

### Week 2: Intent Parser & Prediction Engine
- [ ] Enhance intent parser for prediction recognition
- [ ] Create `MLPredictionAgent` class
- [ ] Implement single prediction functionality
- [ ] Implement batch prediction functionality

### Week 3: Integration & Formatting
- [ ] Create prediction response formatters
- [ ] Integrate prediction handling in enhanced uAgent
- [ ] Update query processing logic
- [ ] Add error handling for prediction scenarios

### Week 4: Testing & Refinement
- [ ] Create comprehensive test suite
- [ ] Test various prediction scenarios
- [ ] Optimize performance and memory usage
- [ ] Documentation and examples

## 🔧 Usage Examples

### Example 1: Train Then Predict
```
User: "Clean and train ML model using https://example.com/churn_data.csv to predict Churn"
Agent: [Executes full workflow, stores model in session]

User: "Predict churn for CustomerID=123, MonthlyCharges=65.5, TotalCharges=1500"
Agent: [Uses stored model to make prediction]
```

### Example 2: Batch Predictions
```
User: "Use the trained model to predict for https://example.com/new_customers.csv"
Agent: [Makes batch predictions, returns downloadable results]
```

### Example 3: Model Analysis
```
User: "What features are most important for churn prediction?"
Agent: [Analyzes stored model, returns feature importance insights]
```

## ✅ Success Criteria

1. **Seamless Workflow**: User can train model and immediately use it for predictions
2. **Session Management**: Model persists within session, expires appropriately
3. **Multiple Prediction Types**: Single predictions, batch predictions, model analysis
4. **Error Handling**: Graceful handling when no model exists or prediction fails
5. **Performance**: Efficient model loading and prediction execution
6. **User Experience**: Clear, helpful responses for all prediction scenarios

## 🚧 Technical Considerations

### Memory Management
- Store only model metadata in session, load H2O model on-demand
- Clean up H2O resources after predictions
- Monitor memory usage for large models

### Error Handling
- Model loading failures
- H2O cluster connectivity issues
- Invalid prediction data formats
- Session expiration scenarios

### Security
- Validate prediction input data
- Sanitize file paths for model loading
- Limit prediction batch sizes

This implementation leverages the existing excellent architecture while adding powerful ML prediction capabilities that feel natural and integrated into the current workflow.

## 🔧 Key Improvements Made

### 1. **Enhanced Model Info Storage**
- ✅ **Rich H2O AutoML Data**: Now stores complete leaderboard, top model metrics, feature importance
- ✅ **Comprehensive Session**: Includes generated code, recommended steps, workflow summary
- ✅ **Total Models Trained**: Tracks how many models were tested in AutoML

### 2. **Proper LLM Intent Parsing**
- ✅ **No More Regex**: Removed keyword-based detection entirely
- ✅ **Smart Intent Recognition**: Uses existing LLM-based intent parser
- ✅ **Single Parse**: Intent parsed once and passed to handlers

### 3. **Streamlined Query Processing**
- ✅ **Commented Out Data Delivery**: Removed unnecessary data delivery handling
- ✅ **Focused on ML**: Query processing now focused on ML prediction workflow
- ✅ **Clean Architecture**: Maintains existing patterns while adding ML capabilities

### 4. **Enhanced Model Analysis**
- ✅ **Comprehensive Answers**: Uses all stored H2O data for detailed analysis
- ✅ **Feature Importance**: Can answer detailed questions about feature importance
- ✅ **Leaderboard Insights**: Can explain why specific models were chosen
- ✅ **Performance Analysis**: Detailed metrics and cross-validation results

### 5. **Simplified Architecture (Key Correction)**
- ✅ **Reuse Existing Schema**: No new `TrainedModelSession` needed - use existing `MLModelingMetrics`
- ✅ **Seamless Integration**: Store `MLModelingMetrics` directly from successful ML training
- ✅ **Consistent with Existing Patterns**: Follows the same session management pattern as cleaned data
- ✅ **Complete Method Implementations**: All helper methods for prediction formatting included 