# Implementation Plan: Copy AI Data Science Team Capabilities

## Overview
Copy specific capabilities and code from `/ai-data-science-team` repository to `/ai-data-science` folder, excluding ML agents, ML tools, SQL data analyst, supervised data analyst, and SQL-database-agent-app.

## Target Components to Copy

### ✅ Core Infrastructure
- [ ] Base tools and utilities (excluding ML tools)
- [ ] Templates and parsers
- [ ] Core orchestration utilities

### ✅ Data Science Agents
- [ ] Data cleaning agent
- [ ] Data loader tools agent  
- [ ] Data visualisation agent (rename from visualization)
- [ ] Data wrangling agent
- [ ] Feature engineering agent

### ✅ DS Agents (Specialized)
- [ ] EDA tools agent
- [ ] Pandas data analyst agent (from multiagents)

### ✅ Examples to Copy
- [ ] data_cleaning_agent.ipynb
- [ ] data_loader_tools_agent.ipynb
- [ ] data_visualization_agent.ipynb (rename to data_visualisation_agent.ipynb)
- [ ] data_wrangling_agent.ipynb
- [ ] feature_engineering_agent.ipynb
- [ ] ds_agents/eda_tools_agent.ipynb
- [ ] multiagents/pandas_data_analyst.ipynb

### ✅ Supporting Files
- [ ] Update requirements.txt with necessary dependencies
- [ ] Update setup.py if needed
- [ ] Create proper __init__.py files for package structure
- [ ] Update README.md (remove branding/fluff)

## Components to EXCLUDE
- ❌ ml_agents/ (entire directory)
- ❌ ML tools from tools/ (h2o.py, mlflow.py) 
- ❌ sql_data_analyst.py from multiagents
- ❌ supervised_data_analyst.py from multiagents
- ❌ sql-database-agent-app from apps/
- ❌ All ML-related examples
- ❌ Business Science branding and promotional content

## Implementation Steps

### Phase 1: Setup and Infrastructure
- [ ] Create core package structure in ai-data-science
- [ ] Copy base utilities and orchestration
- [ ] Copy templates and parsers (excluding ML-related ones)
- [ ] Set up proper __init__.py files

### Phase 2: Core Tools
- [ ] Copy data_loader.py tools
- [ ] Copy dataframe.py tools
- [ ] Copy eda.py tools
- [ ] Update naming conventions (visualization → visualisation)

### Phase 3: Data Science Agents
- [ ] Copy data_cleaning_agent.py
- [ ] Copy data_loader_tools_agent.py
- [ ] Copy data_visualization_agent.py (rename appropriately)
- [ ] Copy data_wrangling_agent.py
- [ ] Copy feature_engineering_agent.py
- [ ] Copy eda_tools_agent.py
- [ ] Copy pandas_data_analyst.py

### Phase 4: Examples and Documentation
- [ ] Copy relevant example notebooks
- [ ] Update example notebooks to match new package structure
- [ ] Remove branding and promotional content
- [ ] Update import statements to match new package name
- [ ] Test examples to ensure they work

### Phase 5: Package Configuration
- [ ] Update requirements.txt
- [ ] Update setup.py
- [ ] Create clean README.md
- [ ] Update .gitignore if needed

### Phase 6: Testing and Validation
- [ ] Test key functionality works
- [ ] Verify naming conventions are consistent
- [ ] Check all imports work correctly
- [ ] Run example notebooks to validate

## Notes
- Ensure all "visualization" references are changed to "visualisation" for consistency
- Remove all Business Science branding and promotional content
- Maintain the same functionality but adapt to ai-data-science package structure
- Keep the same high-quality implementations but clean up unnecessary fluff

## Status
- [x] Plan created
- [x] Implementation started
- [x] Core infrastructure copied (utils, parsers, templates)
- [x] EDA tools copied
- [ ] Agents copied (in progress)
- [ ] Examples copied
- [ ] Package configured
- [ ] Testing completed
- [ ] Implementation complete

## Progress Update
### Completed:
- ✅ Added html.py and matplotlib.py utilities
- ✅ Updated utils __init__.py with new exports
- ✅ Added SQLOutputParser to parsers
- ✅ Created comprehensive eda.py tools with visualise_missing (UK spelling)
- ✅ Updated tools __init__.py to export EDA tools
- ✅ Created ds_agents directory and copied EDAToolsAgent
- ✅ Created multiagents directory and copied PandasDataAnalyst
- ✅ Updated main src __init__.py to export new modules
- ✅ Created example notebooks for data cleaning and EDA tools
- ✅ Updated requirements.txt with all necessary dependencies

### Currently Working On:
- ✅ Copied EDA tools agent from ds_agents
- ✅ Copied pandas data analyst from multiagents
- ✅ Created example notebooks for data cleaning and EDA
- ✅ Updated requirements.txt with necessary dependencies

### Next Steps:
- Copy data_wrangling_agent.py (large file - need to adapt imports)
- Copy feature_engineering_agent.py 
- Copy remaining example notebooks
- Update package configuration and test functionality 