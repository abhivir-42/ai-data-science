# uAgent Test Prompts

## Overview
Test prompts to verify the enhanced data analysis uAgent correctly identifies which workflow steps to execute based on user intent. Each prompt should trigger only the specified agents.

---

## 🧹 **Data Cleaning Only** (Should run: Cleaning=True, FE=False, ML=False)

### Basic Cleaning Requests
```
Clean the dataset. Here is the data: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv
```

```
Preprocess and clean this data: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
```

```
Handle missing values and duplicates in https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv
```

```
Remove outliers and fix data quality issues: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv
```

### Advanced Cleaning Requests
```
Perform comprehensive data cleaning including missing value imputation, duplicate removal, and outlier detection on https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
```

```
Clean and validate the data quality of this dataset: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/car_crashes.csv
```

---

## 🔧 **Feature Engineering Only** (Should run: Cleaning=False, FE=True, ML=False)

### Basic Feature Engineering
```
Engineer features for this dataset: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
```

```
Transform categorical variables in https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv
```

```
Create new features from https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv
```

### Advanced Feature Engineering
```
Encode categorical variables and create interaction features for https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
```

```
Generate datetime features and perform feature selection on https://raw.githubusercontent.com/mwaskom/seaborn-data/master/car_crashes.csv
```

---

## 🤖 **Full ML Pipeline** (Should run: Cleaning=True, FE=True, ML=True)

### Classification Tasks
```
Build a machine learning model to predict species using https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
```

```
Create a classification model to predict survival using https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
```

```
Train a model to classify high vs low tips in https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv
```

### Regression Tasks
```
Build a regression model to predict mpg using https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv
```

```
Create a predictive model for tip amounts using https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv
```

### AutoML Requests
```
Perform complete machine learning analysis on https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv for species prediction
```

```
Run full AutoML pipeline to predict passenger survival: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
```

---

## 🔄 **Mixed Workflows**

### Cleaning + Feature Engineering (Should run: Cleaning=True, FE=True, ML=False)
```
Clean the data and engineer features for https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv
```

```
Preprocess and transform features in https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv
```

### Feature Engineering + ML (Should run: Cleaning=False, FE=True, ML=True)
```
Engineer features and build a model to predict species using https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
```

---

## 🎯 **Edge Cases & Ambiguous Requests**

### Potentially Ambiguous (Test LLM interpretation)
```
Analyze this dataset: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
```

```
Process this data for insights: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv
```

```
Prepare this dataset for analysis: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv
```

### Exploratory Analysis
```
Explore and understand this dataset: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/car_crashes.csv
```

```
Generate insights from this data: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv
```

---

## 🧪 **Performance Testing**

### Small Datasets (Should use 30s runtime)
```
Clean this small dataset: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
```

### Medium Datasets
```
Build a model using this dataset: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
```

---

## 🚫 **Error Handling Tests**

### Invalid URLs
```
Clean this dataset: https://invalid-url.com/nonexistent.csv
```

### Non-CSV URLs
```
Analyze this data: https://www.google.com
```

### Missing URLs
```
Clean the dataset and build a machine learning model
```

---

## 📊 **Expected Results Guide**

| Request Type | Cleaning | Feature Engineering | ML Modeling | Runtime |
|-------------|----------|-------------------|-------------|---------|
| "Clean the dataset" | ✅ | ❌ | ❌ | 30s |
| "Engineer features" | ❌ | ✅ | ❌ | 30-60s |
| "Build ML model" | ✅ | ✅ | ✅ | 30-60s |
| "Clean and engineer" | ✅ | ✅ | ❌ | 30-60s |
| "Analyze dataset" | ? | ? | ? | Depends on LLM |

---

## 🎯 **Testing Instructions**

1. **Send each prompt** to your running uAgent
2. **Check the logs** for the intent parsing results:
   ```
   INFO:src.agents.data_analysis_agent:Executing workflow - Cleaning: X, FE: Y, ML: Z
   ```
3. **Verify** only the expected agents run
4. **Monitor runtime** for performance optimization
5. **Check confidence scores** (should be 0.7+ for clear requests)

---

## 🏆 **Success Criteria**

- ✅ **Cleaning-only requests** should NOT trigger FE or ML
- ✅ **FE-only requests** should NOT trigger cleaning or ML  
- ✅ **ML requests** should trigger all three (cleaning + FE + ML)
- ✅ **High confidence** (0.7+) for clear requests
- ✅ **Appropriate runtime** based on dataset size
- ✅ **No fallback logic** triggered (should see retry attempts if any failures)

Use these prompts to thoroughly test your enhanced uAgent! 🚀 