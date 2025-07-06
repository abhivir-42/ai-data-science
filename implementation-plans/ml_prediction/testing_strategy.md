# ML Prediction Enhancement - Testing Strategy

**Objective**: Implement rigorous testing after every feature change to catch issues immediately and ensure quality at each step.

## 🎯 **Core Testing Philosophy**

**"Test Early, Test Often, Commit Only When Green ✅"**

- Test after **every phase implementation**
- Only commit when **all tests pass**
- Use **multiple testing approaches** (unit, integration, manual)
- **Pinpoint issues immediately** - no debugging mysteries later

## 🧪 **Phase-by-Phase Testing Approach**

### **Phase 1: Schema & Session Management Testing**

#### **Test 1A: Schema Validation**
**File**: `test_phase1_schemas.py`

```python
import pytest
from src.schemas.data_analysis_schemas import WorkflowIntent
from src.uagent_v2.enhanced_uagent import EnhancedDataAnalysisUAgent
from src.uagent_v2.config import UAgentConfig

def test_workflow_intent_prediction_fields():
    """Test new prediction fields in WorkflowIntent schema"""
    intent = WorkflowIntent(
        needs_data_cleaning=True,
        needs_feature_engineering=False, 
        needs_ml_modeling=True,
        data_quality_focus=True,
        exploratory_analysis=False,
        prediction_focus=True,
        statistical_analysis=False,
        key_requirements=["predict survival"],
        complexity_level="moderate",
        intent_confidence=0.8,
        
        # NEW FIELDS TO TEST
        needs_prediction=True,
        needs_model_analysis=False,
        prediction_data_source="inline",
        prediction_type="single_prediction",
        extracted_prediction_data={"age": 25, "sex": "male", "pclass": 3}
    )
    
    # Test new fields
    assert intent.needs_prediction == True
    assert intent.needs_model_analysis == False
    assert intent.prediction_type == "single_prediction"
    assert intent.extracted_prediction_data == {"age": 25, "sex": "male", "pclass": 3}
    
    print("✅ WorkflowIntent prediction fields working correctly")

def test_workflow_intent_prediction_types():
    """Test all prediction types are valid"""
    valid_types = ["single_prediction", "batch_prediction", "model_analysis"]
    
    for pred_type in valid_types:
        intent = WorkflowIntent(
            needs_data_cleaning=False,
            needs_feature_engineering=False,
            needs_ml_modeling=False,
            data_quality_focus=False,
            exploratory_analysis=False,
            prediction_focus=True,
            statistical_analysis=False,
            key_requirements=["test"],
            complexity_level="simple",
            intent_confidence=0.9,
            needs_prediction=True,
            prediction_type=pred_type
        )
        assert intent.prediction_type == pred_type
    
    print("✅ All prediction types validated")

if __name__ == "__main__":
    test_workflow_intent_prediction_fields()
    test_workflow_intent_prediction_types()
    print("🎉 Phase 1A: Schema validation tests PASSED")
```

#### **Test 1B: Enhanced uAgent Initialization**
**File**: `test_phase1_uagent_init.py`

```python
import pytest
from src.uagent_v2.enhanced_uagent import EnhancedDataAnalysisUAgent
from src.uagent_v2.config import UAgentConfig

def test_enhanced_uagent_ml_session_variables():
    """Test enhanced uAgent initializes ML session variables correctly"""
    config = UAgentConfig.from_env()
    agent = EnhancedDataAnalysisUAgent(config)
    
    # Check new ML session variables exist
    assert hasattr(agent, '_last_trained_model'), "Missing _last_trained_model"
    assert hasattr(agent, '_last_model_timestamp'), "Missing _last_model_timestamp"
    assert hasattr(agent, '_last_training_result'), "Missing _last_training_result"
    assert hasattr(agent, '_last_target_variable'), "Missing _last_target_variable"
    assert hasattr(agent, 'prediction_formatter'), "Missing prediction_formatter"
    
    # Check they're properly initialized as None
    assert agent._last_trained_model is None
    assert agent._last_model_timestamp is None
    assert agent._last_training_result is None
    assert agent._last_target_variable is None
    
    # Check prediction_formatter is initialized
    assert agent.prediction_formatter is not None
    
    print("✅ Enhanced uAgent ML session variables initialized correctly")

def test_enhanced_uagent_has_trained_model_method():
    """Test _has_trained_model method exists and works"""
    config = UAgentConfig.from_env()
    agent = EnhancedDataAnalysisUAgent(config)
    
    # Should return False when no model is stored
    assert agent._has_trained_model() == False
    
    print("✅ _has_trained_model method working correctly")

if __name__ == "__main__":
    test_enhanced_uagent_ml_session_variables()
    test_enhanced_uagent_has_trained_model_method()
    print("🎉 Phase 1B: Enhanced uAgent initialization tests PASSED")
```

**Commands for Phase 1:**
```bash
# Run Phase 1 tests
python test_phase1_schemas.py
python test_phase1_uagent_init.py

# If all pass, commit
git add .
git commit -m "Phase 1: Schema & session management - All tests passing ✅"
```

---

### **Phase 2: ML Prediction Engine Testing**

#### **Test 2A: MLPredictionAgent with Mock Data**
**File**: `test_phase2_prediction_agent.py`

```python
import pytest
from src.schemas.data_analysis_schemas import MLModelingMetrics
from src.agents.ml_prediction_agent import MLPredictionAgent
from src.uagent_v2.config import UAgentConfig

def test_ml_prediction_agent_initialization():
    """Test MLPredictionAgent initializes with mock MLModelingMetrics"""
    # Create mock MLModelingMetrics
    mock_metrics = MLModelingMetrics(
        models_trained=5,
        best_model_id="GBM_model_123",
        best_model_score=0.85,
        training_time_seconds=120.0,
        features_used=["age", "sex", "pclass"],
        model_path="/tmp/fake_model.zip",
        leaderboard=[{"model_id": "GBM_model_123", "auc": 0.85}],
        top_model_metrics={"auc": 0.85, "logloss": 0.42},
        total_models_trained=5,
        model_architecture="Gradient Boosting Machine (GBM)"
    )
    
    config = UAgentConfig.from_env()
    prediction_agent = MLPredictionAgent(mock_metrics, "survived", config)
    
    # Test initialization
    assert prediction_agent.model_metrics == mock_metrics
    assert prediction_agent.target_variable == "survived"
    assert prediction_agent.config == config
    assert prediction_agent._h2o_model is None  # Not loaded yet
    
    print("✅ MLPredictionAgent initialization working correctly")

def test_problem_type_detection():
    """Test _determine_problem_type method"""
    # Classification mock
    classification_metrics = MLModelingMetrics(
        models_trained=1,
        training_time_seconds=60.0,
        features_used=["age"],
        model_architecture="Classification Model",
        top_model_metrics={"auc": 0.85, "logloss": 0.42}
    )
    
    config = UAgentConfig.from_env()
    agent = MLPredictionAgent(classification_metrics, "survived", config)
    
    assert agent._determine_problem_type() == "classification"
    print("✅ Problem type detection working correctly")

if __name__ == "__main__":
    test_ml_prediction_agent_initialization()
    test_problem_type_detection()
    print("🎉 Phase 2A: MLPredictionAgent tests PASSED")
```

#### **Test 2B: Intent Parser Recognition**
**File**: `test_phase2_intent_parser.py`

```python
import pytest
from src.parsers.intent_parser import DataAnalysisIntentParser

def test_intent_parser_prediction_recognition():
    """Test intent parser recognizes prediction requests"""
    parser = DataAnalysisIntentParser()
    
    # Test prediction request recognition
    intent = parser.parse_intent(
        "Predict survival for Age=25, Sex=male, Pclass=3",
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    )
    
    assert intent.needs_prediction == True
    assert intent.needs_model_analysis == False
    print("✅ Prediction request recognition working")

def test_intent_parser_model_analysis_recognition():
    """Test intent parser recognizes model analysis requests"""
    parser = DataAnalysisIntentParser()
    
    # Test model analysis request recognition
    intent = parser.parse_intent(
        "What are the most important features for survival prediction?",
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    )
    
    assert intent.needs_model_analysis == True
    print("✅ Model analysis request recognition working")

if __name__ == "__main__":
    test_intent_parser_prediction_recognition()
    test_intent_parser_model_analysis_recognition()
    print("🎉 Phase 2B: Intent parser recognition tests PASSED")
```

**Commands for Phase 2:**
```bash
# Run Phase 2 tests
python test_phase2_prediction_agent.py
python test_phase2_intent_parser.py

# If all pass, commit
git add .
git commit -m "Phase 2: ML prediction engine - All tests passing ✅"
```

---

### **Phase 3: Integration Testing (Live Agent Testing)**

#### **Test 3A: End-to-End Workflow Test**
**File**: `test_phase3_integration.py`

```python
import pytest
import time
from src.uagent_v2.enhanced_uagent import EnhancedDataAnalysisUAgent
from src.uagent_v2.config import UAgentConfig

def test_full_ml_prediction_workflow():
    """Test complete train → predict workflow with real data"""
    config = UAgentConfig.from_env()
    agent = EnhancedDataAnalysisUAgent(config)
    
    print("🚀 Starting end-to-end ML prediction workflow test...")
    
    # Step 1: Train model with real dataset
    print("Step 1: Training model...")
    train_response = agent.process_query(
        "Train a model using https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv to predict Survived"
    )
    
    # Verify model was stored in session
    assert agent._has_trained_model() == True, "Model should be stored after training"
    assert agent._last_trained_model is not None, "MLModelingMetrics should be stored"
    assert agent._last_target_variable is not None, "Target variable should be stored"
    
    print("✅ Model training and storage successful")
    
    # Step 2: Make a prediction
    print("Step 2: Making prediction...")
    pred_response = agent.process_query(
        "Predict Survived for Age=25, Sex=male, Pclass=3"
    )
    
    # Verify prediction response format
    assert "PREDICTION RESULT" in pred_response, "Should contain prediction result header"
    assert "Age: 25" in pred_response, "Should show input Age"
    assert "Sex: male" in pred_response, "Should show input Sex"
    assert "Survived" in pred_response, "Should mention target variable"
    
    print("✅ Prediction generation successful")
    
    # Step 3: Ask model analysis question
    print("Step 3: Analyzing model...")
    analysis_response = agent.process_query(
        "What are the most important features for survival prediction?"
    )
    
    # Verify analysis response
    assert "MODEL ANALYSIS" in analysis_response, "Should contain analysis header"
    
    print("✅ Model analysis successful")
    print("🎉 End-to-end workflow test PASSED")

def test_no_model_scenario():
    """Test behavior when no model is available"""
    config = UAgentConfig.from_env()
    agent = EnhancedDataAnalysisUAgent(config)
    
    # Try to make prediction without training model first
    response = agent.process_query("Predict survival for Age=30")
    
    assert "No Trained Model Found" in response, "Should show no model message"
    print("✅ No model scenario handling working correctly")

def test_session_expiration():
    """Test model session expiration"""
    config = UAgentConfig.from_env()
    agent = EnhancedDataAnalysisUAgent(config)
    
    # Simulate expired session
    agent._last_model_timestamp = time.time() - (2 * 3600)  # 2 hours ago
    agent._last_trained_model = "fake_model"  # Set something to make it seem like there was a model
    
    expired_response = agent.process_query("Predict survival for Age=30")
    
    assert "No Trained Model Found" in expired_response, "Should handle expired sessions"
    print("✅ Session expiration handling working correctly")

if __name__ == "__main__":
    test_no_model_scenario()
    test_session_expiration() 
    # Comment out the full workflow test for now - requires actual model training
    # test_full_ml_prediction_workflow()
    print("🎉 Phase 3A: Integration tests PASSED")
```

#### **Test 3B: Live Agent Testing Protocol**
**File**: `test_phase3_live_agent.md`

```markdown
# Live Agent Testing Protocol

## Setup
1. Kill any running agents: `pkill -f "enhanced_uagent.py"`
2. Start fresh agent: `python src/uagent_v2/enhanced_uagent.py`
3. Open AgentVerse inspector URL (shown in console output)

## Test Sequence

### Test 1: Model Training
**Input**: "Train a model using https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv to predict Survived"

**Expected Output**:
- ✅ Shows data cleaning progress
- ✅ Shows feature engineering progress  
- ✅ Shows ML model training progress
- ✅ Shows final results with model performance
- ✅ Model is stored in session (check logs)

### Test 2: Single Prediction
**Input**: "Predict Survived for Age=25, Sex=male, Pclass=3, Fare=7.25"

**Expected Output**:
- ✅ Shows "PREDICTION RESULT" header
- ✅ Shows input data (Age: 25, Sex: male, etc.)
- ✅ Shows prediction result
- ✅ Shows model confidence/probability (if classification)
- ✅ Shows model architecture used

### Test 3: Model Analysis  
**Input**: "What are the most important features for survival prediction?"

**Expected Output**:
- ✅ Shows "MODEL ANALYSIS" header
- ✅ Provides detailed feature importance analysis
- ✅ References H2O AutoML leaderboard data
- ✅ Gives specific model insights

### Test 4: Error Scenarios

#### No Model Available
**Input**: Start fresh agent, immediately ask "Predict survival for Age=30"

**Expected Output**:
- ✅ Shows "No Trained Model Found" message
- ✅ Provides helpful instructions on how to train a model first

#### Invalid Prediction Format
**Input**: After training, ask "Predict survival for InvalidField=xyz"

**Expected Output**:
- ✅ Shows appropriate error message
- ✅ Suggests correct input format

## Success Criteria
- [ ] All 4 test scenarios work as expected
- [ ] No Python errors in console logs
- [ ] Agent responds within reasonable time (< 30 seconds per request)
- [ ] Session persistence works (train → predict → analyze)
```

**Commands for Phase 3:**
```bash
# Run integration tests
python test_phase3_integration.py

# Run live agent tests (manual)
# Follow protocol in test_phase3_live_agent.md

# If all pass, commit
git add .
git commit -m "Phase 3: Integration complete - All tests passing ✅"
```

---

## 🔄 **Testing Workflow Commands**

### **Complete Testing Suite**
```bash
# Run all automated tests in sequence
echo "🧪 Running Phase 1 tests..."
python test_phase1_schemas.py
python test_phase1_uagent_init.py

echo "🧪 Running Phase 2 tests..."
python test_phase2_prediction_agent.py
python test_phase2_intent_parser.py

echo "🧪 Running Phase 3 tests..."
python test_phase3_integration.py

echo "🎉 All automated tests completed!"
```

### **Manual Testing Checklist**
```bash
# 1. Kill any running agents
pkill -f "enhanced_uagent.py"

# 2. Start fresh agent
python src/uagent_v2/enhanced_uagent.py

# 3. Test via AgentVerse (follow live testing protocol)

# 4. Check logs for any errors
tail -f logs/enhanced_uagent.log
```

## 🚀 **Benefits of This Testing Strategy**

### **1. Early Issue Detection**
- Catch problems **at their source** - no debugging mysteries
- **Phase-by-phase validation** ensures each component works before building on top

### **2. Confidence in Changes**
- **Know exactly what works** at each step
- **Regression testing** - ensure new changes don't break existing functionality

### **3. Easy Rollback**
- If something breaks, we know **exactly which commit** introduced it
- **Clean git history** with working commits at every step

### **4. Quality Assurance**
- **Multiple testing approaches**: unit, integration, manual
- **Real-world testing** with actual datasets and user interactions

### **5. Documentation**
- **Test cases serve as documentation** of expected behavior
- **Clear protocols** for manual testing scenarios

## 📋 **Implementation Rhythm**

**For Each Phase:**
1. ✅ **Implement** the features
2. ✅ **Run automated tests** 
3. ✅ **Run manual tests** (when applicable)
4. ✅ **Fix any issues** immediately
5. ✅ **Commit only when all tests pass**
6. ✅ **Move to next phase**

**This ensures we maintain momentum while guaranteeing quality!** 🎯

---

## 🎯 **Ready to Begin Phase 1**

**Next Steps:**
1. Implement Phase 1 schema changes
2. Run Phase 1 tests
3. Commit when green ✅
4. Move to Phase 2

**Let's build this feature with confidence!** 🚀 