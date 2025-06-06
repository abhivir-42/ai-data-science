# Data Visualization Agent Demos

## Overview

This collection of demos showcases the comprehensive capabilities of the **DataVisualizationAgent**, an AI-powered system that automatically generates professional-quality Plotly visualizations from simple text instructions.

## 🚀 Quick Start

### Prerequisites

1. **Environment Setup**: Ensure you have the virtual environment activated
   ```bash
   source /Users/abhivir42/projects/ai-ds-venv/bin/activate
   ```

2. **API Keys**: Make sure your OpenAI API key is set in the `.env` file
   ```
   OPENAI_API_KEY=sk-proj-...
   ```

3. **Dependencies**: All required packages should be installed (plotly, langchain, etc.)

### Running the Demos

1. **Quick Test** (recommended first):
   ```bash
   python demo_quick_test.py
   ```
   - Verifies the agent works correctly
   - Takes ~30 seconds
   - Creates a simple test chart

2. **Individual Demos**:
   ```bash
   python demo_1_basic_usage.py        # Basic features with sales data
   python demo_2_advanced_features.py  # Advanced features with customer analytics
   ```

3. **Complete Demo Suite** (recommended for presentations):
   ```bash
   python run_all_demos.py
   ```
   - Runs all demos automatically
   - Generates comprehensive summary
   - Takes 2-5 minutes
   - Perfect for presentations

## 📊 Demo Components

### Demo 1: Basic Usage - Sales Data Visualization
**File**: `demo_1_basic_usage.py`

**Purpose**: Showcases core functionality with business-friendly sales data

**Features Demonstrated**:
- ✅ Automatic chart type selection
- ✅ Interactive Plotly visualizations  
- ✅ Code generation and logging
- ✅ Multiple chart types from same dataset

**Scenarios**:
1. **Automatic Chart Selection**: Let AI choose optimal chart type
2. **Scatter Plot with Correlation**: Marketing spend vs sales analysis
3. **Time Series Analysis**: Sales trends over time
4. **Categorical Comparison**: Product performance comparison

**Output**: 4 interactive HTML charts + generated Python code

### Demo 2: Advanced Features - Customer Analytics  
**File**: `demo_2_advanced_features.py`

**Purpose**: Demonstrates enterprise-level capabilities with complex customer data

**Features Demonstrated**:
- ✅ Complex dataset handling (customer churn)
- ✅ Professional-grade visualizations
- ✅ Advanced analytics and insights
- ✅ Comprehensive logging and export

**Scenarios**:
1. **Churn Analysis Dashboard**: Multi-dimensional churn patterns
2. **Revenue Impact Analysis**: Financial correlations with trend lines  
3. **Customer Segmentation**: Multi-dimensional customer analysis
4. **Tenure vs Loyalty**: Advanced analytics with insights

**Output**: 4 professional-grade charts + detailed analytics code

## 🎯 Key Features Highlighted

### Core Capabilities
- **AI-Powered**: Automatically selects optimal chart types based on data analysis
- **Professional Quality**: Generates publication-ready visualizations
- **Code Transparency**: Provides full Python code for every visualization
- **Error Handling**: Built-in retry mechanisms and error recovery

### Enterprise Features  
- **Complex Data Handling**: Works with multi-dimensional business datasets
- **Logging & Export**: Comprehensive logging of all generated code
- **Customizable**: Flexible parameters and instruction handling
- **Scalable**: Handles datasets from simple to enterprise complexity

### Chart Types Supported
- Bar charts, scatter plots, line charts
- Multi-dimensional visualizations
- Time series analysis
- Categorical comparisons
- Correlation analysis with trend lines
- Professional styling and theming

## 📁 Output Structure

After running demos, you'll find:

```
ai-data-science/
├── output/                          # Generated charts
│   ├── demo_1_scenario_1_*.html    # Demo 1 charts
│   ├── demo_2_scenario_1_*.html    # Demo 2 charts
│   └── quick_test_chart.html       # Test chart
├── logs/                           # Generated code
│   ├── demo_1/                     # Demo 1 Python code
│   └── demo_2/                     # Demo 2 Python code
├── demo_summary.json               # Comprehensive metrics
└── *.py                           # Demo scripts
```

## 🎤 Presentation Guide

### For Business Audiences
1. **Start with Quick Test**: Show the agent works in 30 seconds
2. **Demo 1 Highlights**: Focus on ease of use and automatic chart selection
3. **Show Generated Charts**: Open HTML files to demonstrate interactivity
4. **Business Value**: Emphasize time savings and professional quality

### For Technical Audiences  
1. **Show Code Generation**: Highlight the generated Python functions
2. **Demo 2 Advanced Features**: Focus on complex data handling
3. **Error Handling**: Demonstrate retry mechanisms
4. **Logging Capabilities**: Show comprehensive code tracking

### Key Talking Points
- 🤖 **AI-Powered**: "Just describe what you want, get professional charts"
- 🎨 **Professional**: "Publication-ready visualizations automatically"
- 🔧 **Flexible**: "Handles any dataset, any requirement"
- 📝 **Transparent**: "Full code generation, no black boxes"
- 🔄 **Reliable**: "Built-in error handling and retry logic"
- 📈 **Scalable**: "From simple charts to enterprise dashboards"

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Make sure you're in the right directory
   cd /Users/abhivir42/projects/fetch/ai-data-science
   # And virtual environment is active
   source /Users/abhivir42/projects/ai-ds-venv/bin/activate
   ```

2. **API Key Issues**:
   ```bash
   # Check .env file exists and has correct key
   cat .env | grep OPENAI_API_KEY
   ```

3. **Missing Dependencies**:
   ```bash
   pip install plotly langchain-openai python-dotenv
   ```

4. **Permission Issues**:
   ```bash
   # Ensure output directories can be created
   mkdir -p output logs
   ```

### Getting Help

If you encounter issues:
1. Run `python demo_quick_test.py` to verify basic functionality
2. Check the error messages for specific issues
3. Ensure all dependencies are installed
4. Verify API keys are correctly set

## 📈 Demo Metrics

The complete demo suite generates:
- **~8 interactive charts** (HTML format)
- **~8 Python code files** (fully functional)
- **Comprehensive summary** (JSON format)
- **Execution metrics** (timing, success rates)

Perfect for showcasing the agent's capabilities in presentations, proposals, or technical demonstrations.

## 🎉 Next Steps

After running the demos:
1. **Open generated charts** in your browser for interactivity
2. **Review generated code** to understand the AI's approach
3. **Use demo_summary.json** for detailed metrics
4. **Adapt the examples** for your own datasets
5. **Integrate into your workflows** for automated visualization

The DataVisualizationAgent is ready for production use in data science workflows, business intelligence dashboards, and automated reporting systems. 