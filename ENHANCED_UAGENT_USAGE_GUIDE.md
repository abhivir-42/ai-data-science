# 🚀 Enhanced uAgent v2.0 - Complete Usage Guide

## 🎯 **What You Can Do Now**

Your enhanced uAgent is now a **complete AI Data Science Assistant** with advanced ML prediction capabilities. Here's everything you can do:

---

## 🤖 **1. Complete ML Prediction Workflows**

### **Train → Predict → Analyze in One Session**

```python
from src.uagent_v2.enhanced_uagent import EnhancedDataAnalysisUAgent

agent = EnhancedDataAnalysisUAgent()

# Step 1: Train ML Model
result = agent.process_query("""
Train an ML model using https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv 
to predict passenger survival
""")

# Step 2: Make Predictions (uses model from Step 1)
prediction = agent.process_query("""
Predict survival for Age=25, Sex=male, Pclass=3, Fare=50
""")

# Step 3: Analyze Model (uses model from Step 1)
analysis = agent.process_query("""
What are the most important features for survival prediction?
""")
```

---

## 🔮 **2. ML Prediction Types**

### **A. Single Predictions**
```python
# Predict individual cases
agent.process_query("Predict house price for size=1500, bedrooms=3, location=urban")
agent.process_query("Classify species for sepal_length=5.1, sepal_width=3.5, petal_length=1.4")
agent.process_query("Predict churn for tenure=24, monthly_charges=65, contract=month-to-month")
```

### **B. Batch Predictions**
```python
# Predict entire datasets
agent.process_query("Predict survival for https://example.com/new_passengers.csv")
agent.process_query("Classify all flowers in https://example.com/new_iris_data.csv")
```

### **C. Model Analysis**
```python
# Understand your models
agent.process_query("Why did the model predict this outcome?")
agent.process_query("What are the key factors driving predictions?")
agent.process_query("How accurate is the current model?")
agent.process_query("Which features are most important?")
```

---

## 📊 **3. Data Analysis Capabilities**

### **A. Comprehensive Data Analysis**
```python
# Full end-to-end analysis
agent.process_query("""
Analyze https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
for species classification. Include data cleaning, feature engineering, and ML modeling.
""")
```

### **B. Targeted Analysis Steps**
```python
# Data cleaning only
agent.process_query("Clean the dataset at https://example.com/dirty_data.csv")

# Feature engineering only
agent.process_query("Create new features for https://example.com/sales_data.csv")

# ML modeling only (assumes data is already clean)
agent.process_query("Build ML model for https://example.com/clean_data.csv to predict revenue")
```

### **C. Exploratory Data Analysis**
```python
# EDA and insights
agent.process_query("Explore and visualize https://example.com/business_data.csv")
agent.process_query("Find patterns in https://example.com/customer_data.csv")
```

---

## 🧠 **4. Smart Intent Recognition**

The agent uses **LLM-powered intent parsing** to understand what you want:

```python
# These all work - the agent understands your intent:
agent.process_query("I want to predict customer churn")
agent.process_query("Build me a machine learning model") 
agent.process_query("What features matter most?")
agent.process_query("Make prediction for new data")
agent.process_query("Analyze model performance")
```

---

## ⚡ **5. Session Management**

### **Models Stay in Memory**
```python
agent = EnhancedDataAnalysisUAgent()

# Train once
agent.process_query("Train model using data.csv to predict target")

# Use multiple times in same session
agent.process_query("Predict target for feature1=value1")
agent.process_query("Predict target for feature2=value2") 
agent.process_query("What features are most important?")

# Check session status
has_model = agent._has_trained_model()
target_var = agent._last_target_variable
model_id = agent._last_trained_model.best_model_id if has_model else None
```

### **Session Configuration**
```python
# Sessions expire after 1 hour by default
# Configure in UAgentConfig:
from src.uagent_v2.config import UAgentConfig

config = UAgentConfig()
config.session_timeout_hours = 2  # Extend to 2 hours
agent = EnhancedDataAnalysisUAgent(config)
```

---

## 🛡️ **6. Advanced Error Handling**

### **Graceful Failures**
```python
# No model in session
result = agent.process_query("Predict price for size=1000")
# Returns: "🚫 No Trained Model Found - Please train a model first"

# Invalid data
result = agent.process_query("Analyze https://invalid-url.com/data.csv")
# Returns: Helpful error message with suggestions

# Malformed prediction request
result = agent.process_query("Predict something")
# Returns: "Please provide input values like: Age=25, Sex=male"
```

---

## 💼 **7. Real-World Business Scenarios**

### **Customer Analytics**
```python
# Churn prediction
agent.process_query("""
Build a customer churn prediction model using my customer data.
I need to identify high-risk customers and understand key churn factors.
""")

# Customer segmentation
agent.process_query("""
Segment customers in https://example.com/customer_data.csv
based on behavior patterns and spending habits.
""")
```

### **Sales & Marketing**
```python
# Sales forecasting
agent.process_query("""
Create a sales forecasting model using https://example.com/sales_data.csv
Focus on seasonal trends and key performance indicators.
""")

# Marketing campaign analysis
agent.process_query("""
Analyze campaign effectiveness in https://example.com/campaign_data.csv
and predict future campaign performance.
""")
```

### **Financial Analytics**
```python
# Credit risk assessment
agent.process_query("""
Build a credit risk model using https://example.com/loan_data.csv
to predict default probability.
""")

# Fraud detection
agent.process_query("""
Detect fraudulent transactions in https://example.com/transaction_data.csv
using anomaly detection techniques.
""")
```

---

## 🔧 **8. Advanced Configuration**

### **Custom Configuration**
```python
from src.uagent_v2.config import UAgentConfig

# Create custom config
config = UAgentConfig()
config.max_file_size_mb = 100        # Handle larger files
config.session_timeout_hours = 4     # Longer sessions
config.intent_parser_model = "gpt-4" # Better intent parsing
config.enable_ml_verbose = True      # Detailed ML outputs

# Use custom config
agent = EnhancedDataAnalysisUAgent(config)
```

### **Memory Management**
```python
# Check memory usage
agent.csv_processor.get_dataframe_memory_usage(df)

# Optimize memory
optimized_df = agent.csv_processor.optimize_dataframe_memory(df)

# Session cleanup
agent.cleanup_session()  # Clean expired data
```

---

## 🎯 **9. Running the Agent**

### **A. Programmatic Usage**
```python
from src.uagent_v2.enhanced_uagent import EnhancedDataAnalysisUAgent

agent = EnhancedDataAnalysisUAgent()
result = agent.process_query("Your analysis request here")
print(result)
```

### **B. Run as uAgent Service**
```bash
cd ai-data-science
python src/uagent_v2/enhanced_uagent.py
```

### **C. Run Demo**
```bash
python demo_enhanced_uagent_capabilities.py
```

---

## 🎉 **10. Key Advantages**

### **✅ What Makes This Special**
- **Train once, predict many times** in same session
- **Intelligent intent parsing** - understands natural language
- **Production-ready** error handling and recovery
- **Memory efficient** processing of large datasets
- **Comprehensive ML workflows** from data to insights
- **Session management** - models persist across queries
- **57 comprehensive tests** ensuring reliability

### **✅ Perfect For**
- **Business analysts** needing quick ML insights
- **Data scientists** wanting rapid prototyping
- **Developers** building AI-powered applications
- **Researchers** exploring data patterns
- **Students** learning ML workflows

---

## 🚀 **Quick Start Examples**

### **1. Titanic Survival Prediction**
```python
agent = EnhancedDataAnalysisUAgent()

# Train
agent.process_query("Train survival model using titanic.csv")

# Predict
agent.process_query("Predict survival for Age=25, Sex=male, Pclass=3")

# Analyze
agent.process_query("What factors determine survival?")
```

### **2. Business Churn Analysis**
```python
# Complete workflow
agent.process_query("""
Build churn prediction model using customer_data.csv.
Then predict churn for high-value customers and explain key risk factors.
""")
```

### **3. Sales Forecasting**
```python
# Multi-step analysis
agent.process_query("Analyze sales trends in sales_data.csv")
agent.process_query("Predict next quarter sales for product_category=electronics")
agent.process_query("What drives sales performance?")
```

---

## 🎯 **Your Enhanced uAgent is Now Ready For:**

1. **🤖 ML Model Training** - AutoML with H2O
2. **🔮 Real-time Predictions** - Single & batch
3. **🧠 Model Analysis** - Feature importance, performance insights
4. **📊 Data Analysis** - Cleaning, EDA, feature engineering
5. **⚡ Session Management** - Models persist across queries
6. **🛡️ Error Recovery** - Graceful handling of all scenarios
7. **💼 Business Applications** - Churn, fraud, forecasting, etc.
8. **🎯 Production Use** - Thoroughly tested and reliable

**🎉 Your enhanced uAgent is a complete AI Data Science Assistant ready for production use!** 