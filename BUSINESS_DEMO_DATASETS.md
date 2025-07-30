# 🎯 Business & Scientific Demo Datasets for Enhanced uAgent v2.0

## 📊 **CURATED DATASETS FOR COMPELLING DEMONSTRATIONS**

---

## 💰 **BUSINESS & FINANCE**

### 🏠 **1. House Price Prediction (Boston Housing)**
**Dataset**: `https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv`
**Business Value**: Real estate investment decisions, property valuation

#### **Training Prompt:**
```
Train ML model using https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv to predict medv
```

#### **Prediction Prompts:**
```
What would be the house price for: rooms=6, age=50, distance=3.5, crime_rate=0.1?

Predict house value for a property with 7 rooms, 20 years old, near the city center

Calculate expected price for: rooms=5, age=10, lstat=15, ptratio=18
```

#### **Analysis Prompts:**
```
What factors most influence house prices in Boston?

How accurate is the model for predicting real estate values?

Which neighborhoods should investors target based on the model?
```

---

### 💳 **2. Credit Card Fraud Detection**
**Dataset**: `https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv`
**Business Value**: Financial security, fraud prevention

#### **Training Prompt:**
```
Train ML model using https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv to predict Class
```

#### **Prediction Prompts:**
```
Is this transaction fraudulent: V1=-1.5, V2=2.3, V3=-0.8, Amount=149.62?

Predict fraud risk for transaction: V4=1.2, V5=-0.5, V10=2.1, Amount=89.75

Analyze these transactions for fraud probability:
- Amount=50, V1=1.1, V2=-2.3
- Amount=1200, V1=-0.8, V2=3.2
```

#### **Analysis Prompts:**
```
What transaction patterns indicate fraud?

How can banks use this model to reduce false positives?

What's the model's precision in detecting actual fraud cases?
```

---

### 📈 **3. Customer Churn Prediction**
**Dataset**: `https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv`
**Business Value**: Customer retention, marketing strategy

#### **Training Prompt:**
```
Train ML model using https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv to predict Churn
```

#### **Prediction Prompts:**
```
Will this customer churn: tenure=12, MonthlyCharges=70, Contract=Month-to-month, InternetService=Fiber optic?

Predict churn risk for customer: tenure=36, MonthlyCharges=45, Contract=Two year, PaymentMethod=Credit card

Which customers are most likely to cancel:
- tenure=3, MonthlyCharges=85, Contract=Month-to-month
- tenure=24, MonthlyCharges=55, Contract=One year
```

#### **Analysis Prompts:**
```
What factors lead to customer churn?

How can the company reduce churn rates?

What's the cost impact of the top churn risk factors?
```

---

## 🏥 **HEALTHCARE & SCIENCE**

### 💊 **4. Diabetes Prediction**
**Dataset**: `https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv`
**Business Value**: Healthcare outcomes, preventive medicine

#### **Training Prompt:**
```
Train ML model using https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv to predict Outcome
```

#### **Prediction Prompts:**
```
Does this patient have diabetes: Glucose=148, BMI=33.6, Age=50, BloodPressure=72?

Predict diabetes risk for patient: Glucose=120, BMI=28.5, Age=45, Pregnancies=2

Assess diabetes probability for these patients:
- Glucose=180, BMI=35, Age=60
- Glucose=95, BMI=22, Age=30
```

#### **Analysis Prompts:**
```
What are the strongest predictors of diabetes?

How can healthcare providers use this for early intervention?

What lifestyle factors should patients focus on?
```

---

### 🧬 **5. Wine Quality Assessment**
**Dataset**: `https://raw.githubusercontent.com/plotly/datasets/master/winequality-red.csv`
**Business Value**: Quality control, product optimization

#### **Training Prompt:**
```
Train ML model using https://raw.githubusercontent.com/plotly/datasets/master/winequality-red.csv to predict quality
```

#### **Prediction Prompts:**
```
What quality rating for wine with: alcohol=12.5, acidity=0.7, pH=3.2, sulphates=0.6?

Predict wine quality for: alcohol=11, volatile acidity=0.5, citric acid=0.3, residual sugar=2.5

Rate these wine compositions:
- alcohol=13, acidity=0.8, pH=3.1
- alcohol=9.5, acidity=0.4, pH=3.5
```

#### **Analysis Prompts:**
```
Which chemical properties most affect wine quality?

How can winemakers optimize their production process?

What's the ideal alcohol content for high-quality wine?
```

---

## 🚗 **AUTOMOTIVE & TRANSPORTATION**

### ⛽ **6. Car Fuel Efficiency (Auto MPG)**
**Dataset**: `https://raw.githubusercontent.com/plotly/datasets/master/auto-mpg.csv`
**Business Value**: Environmental impact, cost efficiency

#### **Training Prompt:**
```
Train ML model using https://raw.githubusercontent.com/plotly/datasets/master/auto-mpg.csv to predict mpg
```

#### **Prediction Prompts:**
```
What's the MPG for car: cylinders=4, displacement=120, horsepower=85, weight=2500?

Predict fuel efficiency for vehicle: cylinders=6, displacement=200, horsepower=150, weight=3200

Compare fuel efficiency for these cars:
- 4 cylinders, 100hp, 2200 lbs
- 8 cylinders, 300hp, 4000 lbs
```

#### **Analysis Prompts:**
```
What car features most impact fuel efficiency?

How can manufacturers improve MPG ratings?

What's the trade-off between power and efficiency?
```

---

### 🚖 **7. Taxi Tip Prediction (NYC)**
**Dataset**: `https://raw.githubusercontent.com/plotly/datasets/master/tips.csv`
**Business Value**: Service optimization, revenue forecasting

#### **Training Prompt:**
```
Train ML model using https://raw.githubusercontent.com/plotly/datasets/master/tips.csv to predict tip
```

#### **Prediction Prompts:**
```
What tip for: total_bill=45, size=3, time=Dinner, day=Sat?

Predict tip amount for bill=$28, party of 2, lunch on Friday

Estimate tips for these scenarios:
- $60 dinner, 4 people, Saturday
- $18 lunch, 2 people, Tuesday
```

#### **Analysis Prompts:**
```
When do customers tip the most?

How can restaurants maximize tip revenue?

What service factors influence tipping behavior?
```

---

## 📚 **EDUCATION & SOCIAL SCIENCE**

### 🎓 **8. Student Performance Prediction**
**Dataset**: `https://raw.githubusercontent.com/AashitaK/A-Step-Towards-Machine-Learning/master/student-mat.csv`
**Business Value**: Educational outcomes, intervention strategies

#### **Training Prompt:**
```
Train ML model using https://raw.githubusercontent.com/AashitaK/A-Step-Towards-Machine-Learning/master/student-mat.csv to predict G3
```

#### **Prediction Prompts:**
```
What final grade for student: studytime=3, failures=0, absences=2, G1=15, G2=16?

Predict performance for student: age=17, studytime=2, failures=1, Medu=3, Fedu=4

Forecast grades for these students:
- studytime=4, failures=0, absences=0
- studytime=1, failures=2, absences=15
```

#### **Analysis Prompts:**
```
What factors most predict student success?

How can schools identify at-risk students early?

What interventions would be most effective?
```

---

### 💰 **9. Employee Salary Prediction**
**Dataset**: `https://raw.githubusercontent.com/plotly/datasets/master/salaries.csv`
**Business Value**: HR strategy, compensation planning

#### **Training Prompt:**
```
Train ML model using https://raw.githubusercontent.com/plotly/datasets/master/salaries.csv to predict salary
```

#### **Prediction Prompts:**
```
What salary for: years_experience=5, education_level=Master, department=Engineering?

Predict compensation for employee: experience=8, education=PhD, location=SF, department=Data Science

Estimate salaries for these roles:
- 3 years experience, Bachelor's, Marketing
- 10 years experience, Master's, Finance
```

#### **Analysis Prompts:**
```
What factors drive salary differences?

How can companies ensure fair compensation?

What's the value of additional education or experience?
```

---

## 🌍 **ENVIRONMENTAL & CLIMATE**

### 🌡️ **10. Air Quality Prediction**
**Dataset**: `https://raw.githubusercontent.com/plotly/datasets/master/air_quality.csv`
**Business Value**: Public health, environmental policy

#### **Training Prompt:**
```
Train ML model using https://raw.githubusercontent.com/plotly/datasets/master/air_quality.csv to predict AQI
```

#### **Prediction Prompts:**
```
What air quality for: temperature=25, humidity=60, wind_speed=10, pressure=1013?

Predict AQI for conditions: temp=30, humidity=80, wind=5, CO=1.2, NO2=45

Forecast air quality for these scenarios:
- Hot day: temp=35, humidity=70, wind=3
- Windy day: temp=20, humidity=50, wind=20
```

#### **Analysis Prompts:**
```
What weather conditions worsen air quality?

How can cities improve air quality management?

What's the impact of different pollutants?
```

---

## 🎯 **DEMONSTRATION STRATEGIES**

### 🚀 **Quick Business Impact Demo (5 min)**
1. **House Prices**: Train model + predict property value
2. **Customer Churn**: Show business cost implications
3. **Fraud Detection**: Demonstrate real-time risk assessment

### 🔬 **Scientific Analysis Demo (10 min)**
1. **Diabetes Prediction**: Healthcare outcomes
2. **Wine Quality**: Quality control optimization
3. **Student Performance**: Educational interventions

### 💼 **Executive Strategy Demo (3 min)**
1. **Customer Churn**: Direct revenue impact
2. **Fraud Detection**: Risk mitigation
3. **Employee Salaries**: HR strategy optimization

---

## 💡 **PRO TIPS FOR DEMONSTRATIONS**

### 🎯 **Choose Based on Audience:**
- **Finance/Banking**: Credit fraud, house prices
- **Healthcare**: Diabetes, patient outcomes
- **Retail/Service**: Customer churn, tips
- **Manufacturing**: Quality control (wine)
- **HR/Management**: Employee salaries, performance
- **Government/Policy**: Air quality, public health

### 🚀 **Compelling Value Propositions:**
- **House Prices**: "Save thousands on real estate decisions"
- **Fraud Detection**: "Prevent million-dollar losses"
- **Customer Churn**: "Increase retention by 15%"
- **Diabetes**: "Enable early intervention, save lives"
- **Air Quality**: "Protect public health"

### 📊 **Highlight Business Metrics:**
- **ROI**: Show cost savings from predictions
- **Risk Reduction**: Demonstrate fraud/churn prevention
- **Efficiency**: Time saved in decision-making
- **Accuracy**: Compare to manual/traditional methods

---

## 🎉 **READY-TO-USE DEMO SCRIPTS**

### 🏠 **Real Estate Investment Scenario:**
```
"Let's help a real estate investor make data-driven decisions..."

Train ML model using https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv to predict medv

"Now let's evaluate a specific property..."

What would be the house price for: rooms=6, age=50, distance=3.5, crime_rate=0.1?

"Let's analyze what drives property values..."

What factors most influence house prices in Boston?
```

### 💳 **Banking Security Scenario:**
```
"Let's build a fraud detection system for a bank..."

Train ML model using https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv to predict Class

"Now let's check a suspicious transaction..."

Is this transaction fraudulent: V1=-1.5, V2=2.3, V3=-0.8, Amount=149.62?

"What patterns should the bank watch for?"

What transaction patterns indicate fraud?
```

---

**These datasets and prompts will showcase the Enhanced uAgent v2.0's capabilities across diverse, high-value business and scientific applications!** 🚀 