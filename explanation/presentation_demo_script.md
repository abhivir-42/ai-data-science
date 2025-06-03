# AI Data Science Agents Demo Script
## 7-8 Minute Presentation Guide

### Overview
This demo showcases **AI-powered data science agents** that can:
- **Load and understand** any dataset automatically
- **Clean and prepare** data using AI-driven analysis
- **Create professional visualizations** from simple text instructions
- **Enable human-in-the-loop** decision making for critical steps

---

## 🎯 Demo Structure (7-8 minutes)

### **Minute 1-2: Setup & Introduction**
### **Minute 3-4: Data Loading & Cleaning Pipeline**
### **Minute 5-6: Visualization Agent Demo**
### **Minute 7-8: Human-in-the-Loop & Wrap-up**

---

## 🚀 Pre-Demo Setup

**Before the presentation, run these commands:**

```bash
# Activate environment
source /Users/abhivir42/projects/ai-ds-venv/bin/activate

# Navigate to project
cd /Users/abhivir42/projects/fetch/ai-data-science

# Clean up previous demo outputs (optional)
rm -rf output/* logs/*

# Verify API keys
echo "Checking API configuration..."
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('✅ API Key loaded:' if os.getenv('OPENAI_API_KEY') else '❌ API Key missing')"
```

---

## 📋 DEMO SCRIPT

### **MINUTE 1-2: INTRODUCTION & QUICK VERIFICATION**

**🎤 SAY:** *"Today I'll show you AI agents that can understand, clean, and visualize any dataset with simple English instructions. Let's start with a quick verification that everything works."*

**💻 COMMAND:**
```bash
# Quick verification - 30 seconds
python demo_quick_test.py
```

**🎤 EXPLAIN WHILE RUNNING:** 
- *"This agent is analyzing sample data and creating a visualization"*
- *"Notice it's automatically choosing the best chart type"*
- *"The agent generates both the chart AND the Python code"*

**📊 SHOW OUTPUT:**
- Open the generated HTML file: `open output/quick_test_chart.html`
- **🎤 SAY:** *"Interactive, professional-quality chart in 30 seconds from raw data"*

---

### **MINUTE 3-4: DATA LOADING & CLEANING PIPELINE**

**🎤 SAY:** *"Now let's see how these agents work together to process messy, real-world data. I'll demonstrate with a customer dataset that has missing values and inconsistencies."*

**💻 COMMAND:**
```bash
# Combined data loading and cleaning pipeline
python examples/combined_workflow_example.py
```

**🎤 EXPLAIN WHILE RUNNING:**
- *"First agent: Automatically loading and understanding the data structure"*
- *"Second agent: AI-powered data cleaning - analyzing patterns and deciding how to handle missing values"*
- *"The AI is generating custom Python functions for this specific dataset"*

**📊 SHOW REAL-TIME:**
```bash
# While the above is running, show the live logs
tail -f logs/data_cleaning_*.log
```

**🎤 KEY POINTS:**
- *"The agents understand data context - they know not to delete valuable information"*
- *"Conservative cleaning approach - imputation over deletion"*
- *"Every step is logged and reproducible"*

---

### **MINUTE 5-6: VISUALIZATION AGENT SHOWCASE**

**🎤 SAY:** *"Now with clean data, let's create professional visualizations with simple English requests."*

**💻 COMMAND:**
```bash
# Run comprehensive visualization demo
python demo_1_basic_usage.py
```

**🎤 EXPLAIN SCENARIOS:**

**Scenario 1:** *"Just asking for 'the best chart' - AI selects optimal visualization"*

**Scenario 2:** *"Asking for correlation analysis - AI adds trend lines and statistics"*

**Scenario 3:** *"Time series request - AI creates interactive timeline"*

**Scenario 4:** *"Product comparison - AI chooses appropriate categorical chart"*

**📊 LIVE DEMONSTRATION:**
```bash
# Open generated charts as they're created
open output/demo_1_scenario_1_*.html
open output/demo_1_scenario_2_*.html
open output/demo_1_scenario_3_*.html
open output/demo_1_scenario_4_*.html
```

**🎤 HIGHLIGHT:**
- *"Each chart is fully interactive - zoom, hover, filter"*
- *"Professional styling applied automatically"*
- *"Complete Python code generated for each visualization"*

---

### **MINUTE 7-8: HUMAN-IN-THE-LOOP DEMONSTRATION**

**🎤 SAY:** *"For critical business decisions, we can enable human oversight. Let me show you the human-in-the-loop feature."*

**💻 COMMAND:**
```bash
# Create a special human-in-the-loop demo
cat > human_in_loop_demo.py << 'EOF'
"""Human-in-the-loop demonstration"""
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.agents.data_cleaning_agent import DataCleaningAgent

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# Load sample data with issues
df = pd.read_csv("examples/sample_data.csv")
print(f"📊 Original data: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"❌ Missing values: {df.isna().sum().sum()}")
print(f"🔍 Duplicates: {df.duplicated().sum()}")

# Create agent with human-in-the-loop enabled
agent = DataCleaningAgent(
    model=llm,
    log=True,
    human_in_the_loop=True,  # This is the key!
    log_path="./demo_logs"
)

print("\n🤖 Agent will now analyze and ask for approval...")
agent.invoke_agent(
    data_raw=df,
    user_instructions="Clean this customer data conservatively. Ask me before removing any data."
)

cleaned_df = agent.get_data_cleaned()
print(f"\n✅ Cleaned data: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns")
print(f"✅ Missing values after cleaning: {cleaned_df.isna().sum().sum()}")
EOF

# Run the human-in-the-loop demo
python human_in_loop_demo.py
```

**🎤 INTERACTION DEMO:**
*When prompted, show different responses:*
- *"Type 'yes' to approve the AI's plan"*
- *"Type 'modify: use mean instead of median' to request changes"*
- *"Type 'no' to reject and ask for alternatives"*

**🎤 SAY:** *"This ensures AI decisions align with business requirements while maintaining efficiency."*

---

## 🎯 ADVANCED COMMANDS FOR DEEPER DEMO

### **If you have extra time or technical audience:**

**💻 Show Generated Code:**
```bash
# Display the generated cleaning function
cat logs/data_cleaning_*/data_cleaner_function.py | head -30
```

**💻 Show Agent Architecture:**
```bash
# Demonstrate the supervisor agent pattern
python examples/supervisor_agent_example.py
```

**💻 Production Pipeline Demo:**
```bash
# Show production-ready features
python examples/production_pipeline_example.py --file examples/sample_data.csv
```

---

## 📊 DEMONSTRATION TALKING POINTS

### **🎯 Business Value**
- **"Zero coding required"** - English instructions only
- **"Professional results"** - Publication-ready visualizations
- **"Intelligent decisions"** - AI understands data context
- **"Full transparency"** - Every step is logged and reproducible
- **"Human oversight"** - Critical decisions can be reviewed

### **🔧 Technical Highlights**
- **"Multi-agent architecture"** - Specialized agents for different tasks
- **"LangChain integration"** - Built on enterprise-grade AI framework
- **"Plotly visualizations"** - Industry-standard interactive charts
- **"Error handling"** - Robust retry logic and graceful failures
- **"Extensible design"** - Easy to add new capabilities

### **🚀 Use Cases**
- **"Automated reporting"** - Transform raw data into insights daily
- **"Data exploration"** - Quickly understand new datasets
- **"Business intelligence"** - Interactive dashboards from text
- **"Data quality"** - Intelligent cleaning and validation
- **"Research support"** - Academic and industry research workflows

---

## 🔧 TROUBLESHOOTING DURING DEMO

### **If something fails:**

**💻 Backup Commands:**
```bash
# Quick fallback demo
python examples/minimal_cleaning_example.py

# Simple visualization
python -c "
from src.agents.data_visualization_agent import DataVisualizationAgent
import pandas as pd
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model='gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
agent = DataVisualizationAgent(model=llm)

df = pd.DataFrame({'x': [1,2,3,4,5], 'y': [2,4,6,8,10]})
agent.invoke_agent(data=df, user_instructions='Create a simple line chart')
print('Chart saved to:', agent.get_plot_path())
"
```

### **Reset Everything:**
```bash
# If needed, clean slate
rm -rf output/* logs/* demo_logs/*
source /Users/abhivir42/projects/ai-ds-venv/bin/activate
cd /Users/abhivir42/projects/fetch/ai-data-science
```

---

## 🎤 CLOSING STATEMENTS

**🎯 30-Second Summary:**
*"These AI agents transform data science from hours of coding to minutes of conversation. They understand your data, make intelligent decisions, and create professional results - all while keeping humans in control of critical choices. This is the future of data analysis: accessible, intelligent, and transparent."*

**📞 Call to Action:**
*"Ready to try this with your own data? Let's discuss how these agents can accelerate your data science workflows."*

---

## 📋 POST-DEMO COMMANDS

**💻 Cleanup and Organize:**
```bash
# Create demo summary
python -c "
import os
import json
from datetime import datetime

summary = {
    'demo_date': datetime.now().isoformat(),
    'charts_generated': len([f for f in os.listdir('output') if f.endswith('.html')]),
    'code_files': len([f for f in os.listdir('logs') if f.endswith('.py')]),
    'status': 'completed'
}

with open('demo_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('Demo completed successfully!')
print(f\"Charts: {summary['charts_generated']}\")
print(f\"Code files: {summary['code_files']}\")
"

# Archive demo outputs
mkdir -p demo_archive/$(date +%Y%m%d_%H%M%S)
cp -r output/* logs/* demo_archive/$(date +%Y%m%d_%H%M%S)/
echo "Demo outputs archived successfully"
```

---

## 🎯 PRESENTER NOTES

### **Timing Guide:**
- **Setup verification: 1 minute**
- **Data pipeline: 2 minutes**
- **Visualization showcase: 2 minutes**
- **Human-in-the-loop: 1.5 minutes**
- **Wrap-up & questions: 1.5 minutes**

### **Key Demo Files:**
- `demo_quick_test.py` - Verification
- `examples/combined_workflow_example.py` - Main pipeline
- `demo_1_basic_usage.py` - Visualization showcase
- `human_in_loop_demo.py` - Interactive decision making

### **Backup Plans:**
- Have pre-generated charts ready in case of network issues
- Screenshots of key outputs
- Recorded demo video as ultimate fallback

**Success metrics:** Professional charts generated, clean data produced, code transparency demonstrated, human oversight shown. 