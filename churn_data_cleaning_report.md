# Churn Data Cleaning Report

**Generated on:** 2025-04-30 19:06:07

## Original Dataset Summary

- **Number of records:** 40
- **Number of columns:** 18
- **Missing values:** 3 (0.42%)

### Column Datatypes

| Column | Type |
|--------|------|
| customer_id | object |
| gender | object |
| age | int64 |
| tenure | int64 |
| phone_service | object |
| multiple_lines | object |
| internet_service | object |
| online_security | object |
| online_backup | object |
| tech_support | object |
| streaming_tv | object |
| streaming_movies | object |
| contract | object |
| paperless_billing | object |
| payment_method | object |
| monthly_charges | float64 |
| total_charges | float64 |
| churn | object |

### Missing Values by Column

| Column | Missing Values |
|--------|---------------|
| paperless_billing | 1 |
| monthly_charges | 2 |

### First 5 rows of original data

```
  customer_id  gender  age  tenure phone_service    multiple_lines internet_service online_security online_backup tech_support streaming_tv streaming_movies        contract paperless_billing             payment_method  monthly_charges  total_charges churn
0  7590-VHVEG  Female   37       1           Yes                No      Fiber optic              No            No           No           No               No  Month-to-month               Yes           Electronic check            29.85          29.85    No
1  5575-GNVDE    Male   59      34           Yes                No              DSL             Yes            No           No           No               No        One year                No               Mailed check            56.95        1889.50    No
2  3668-QPYBK    Male   41       2           Yes                No              DSL             Yes           Yes           No           No               No  Month-to-month               Yes               Mailed check            53.85         108.15   Yes
3  7795-CFOCW    Male   56      45            No  No phone service              DSL             Yes            No          Yes          Yes               No        One year                No  Bank transfer (automatic)            42.30        1840.75    No
4  9237-HQITU  Female   33       2           Yes                No      Fiber optic              No            No           No           No               No  Month-to-month               Yes           Electronic check            70.70         151.65   Yes
```

## Data Cleaning Process

### Recommended Cleaning Steps

# Recommended Data Cleaning Steps:

Given the dataset characteristics and user instructions, here is a series of steps for cleaning and preprocessing the data:

1. **Remove Duplicate Customer Records**: Since the dataset might contain duplicate entries, start by checking and removing any duplicate rows based on the `customer_id` column as it should be unique for each customer.

   ```python
   df = df.drop_duplicates(subset=['customer_id'], keep='first')
   ```

2. **Impute Missing Numerical Values**: For columns with numerical data (e.g., `monthly_charges`), check for missing values and impute them using an appropriate measure. Given user instructions, we can use either mean or median depending on the data distribution of each numerical column to avoid skewing the data.

   ```python
   # Assuming 'monthly_charges' misses some values. 
   df['monthly_charges'] = df['monthly_charges'].fillna(df['monthly_charges'].mean())
   ```

3. **Impute Missing Categorical Values**: For categorical data with missing values, fill them with either "Unknown" or the most frequent value in the column, as per user instructions. Whether to use "Unknown" or the most frequent value can depend on the significance of the missing data and the column context.

   ```python
   df['paperless_billing'] = df['paperless_billing'].fillna('Unknown') # or df['paperless_billing'].mode()[0] if frequent value is preferred
   ```

4. **Handle Outliers in Numerical Columns**: Identify and handle outliers, especially in closely watched columns like `age`, `tenure`, `monthly_charges`, and `total_charges`. Outliers can be defined as those values that fall below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.

   ```python
   Q1 = df['monthly_charges'].quantile(0.25)
   Q3 = df['monthly_charges'].quantile(0.75)
   IQR = Q3 - Q1
   df = df[~((df['monthly_charges'] < (Q1 - 1.5 * IQR)) |(df['monthly_charges'] > (Q3 + 1.5 * IQR)))]
   ```

5. **Converting Data Types to Appropriate Formats**: Ensure all columns are of the appropriate data type for analysis. For instance, `customer_id` should be of type string, numerical features like `age`, `tenure`, `monthly_charges`, and `total_charges` should be of type float or int, and categorical features should be of type category.

   ```python
   df['customer_id'] = df['customer_id'].astype(str)
   df['age'] = df['age'].astype(int)
   df['gender'] = df['gender'].astype('category')
   ```

6. **Ensure Customer IDs are Valid and Consistent**: Validating `customer_id` formats or consistency might involve checking for, and removing any unusual or non-conforming IDs. This step would require specific criteria for what makes an ID valid.

   ```python
   # Example: if valid IDs are known to have a specific format, e.g., XXXX-XXXXX
   df = df[df['customer_id'].str.match(r'^\d{4}-\w{4}$')]
   ```

7. **Additional Cleaning/Validation Steps**: Given the dataset summary, it appears all customer_id values are unique, and there are no overly broad instructions for further cleaning. However, it's beneficial to review the data for any inconsistencies not caught in earlier steps, such as ensuring that customers with "No internet service" align with expected values in related service columns.

These steps are designed to provide a systematic approach to cleaning and preprocessing the dataset based on its characteristics and user instructions. Adjustments might be needed as actual data conditions and analysis goals become clearer.

### Generated Cleaning Function

```python
# ----------------------------------------------------------------------
# Code generated by AI agent: data_cleaning_agent
# ----------------------------------------------------------------------

def data_cleaner(data_raw):
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer

    # Step 1: Remove Duplicate Customer Records
    data_cleaned = data_raw.drop_duplicates(subset=['customer_id'], keep='first')

    # Step 2: Impute Missing Numerical Values
    # For 'monthly_charges'
    monthly_charges_imputer = SimpleImputer(strategy='mean')
    data_cleaned['monthly_charges'] = monthly_charges_imputer.fit_transform(data_cleaned[['monthly_charges']]).ravel()
    
    # Assuming similar steps might be needed for 'total_charges' if it had missing values
    # Uncomment the following code if necessary
    # total_charges_imputer = SimpleImputer(strategy='mean')
    # data_cleaned['total_charges'] = total_charges_imputer.fit_transform(data_cleaned[['total_charges']]).ravel()
    
    # Step 3: Impute Missing Categorical Values
    # For 'paperless_billing', using the most frequent value because it's a binary option with a clear majority
    paperless_billing_imputer = SimpleImputer(strategy='most_frequent')
    data_cleaned['paperless_billing'] = paperless_billing_imputer.fit_transform(data_cleaned[['paperless_billing']]).ravel()
    
    # Step 4: Handle Outliers in Numerical Columns for 'monthly_charges'
    Q1 = data_cleaned['monthly_charges'].quantile(0.25)
    Q3 = data_cleaned['monthly_charges'].quantile(0.75)
    IQR = Q3 - Q1
    data_cleaned = data_cleaned[~((data_cleaned['monthly_charges'] < (Q1 - 1.5 * IQR)) |(data_cleaned['monthly_charges'] > (Q3 + 1.5 * IQR)))]
    
    # Step 5: Converting Data Types to Appropriate Formats
    data_cleaned['customer_id'] = data_cleaned['customer_id'].astype(str)
    data_cleaned['age'] = data_cleaned['age'].astype(int)
    data_cleaned['gender'] = data_cleaned['gender'].astype('category')
    data_cleaned['phone_service'] = data_cleaned['phone_service'].astype('category')
    data_cleaned['paperless_billing'] = data_cleaned['paperless_billing'].astype('category')
    data_cleaned['churn'] = data_cleaned['churn'].astype('category')
    # And so on for other columns requiring type conversion

    # Step 6: Ensure Customer IDs are Valid and Consistent
    # Since example regex was given, I'll use it, but adjust according to your dataset's actual ID format
    # Uncomment the line below if there's a specific pattern to validate.
    # data_cleaned = data_cleaned[data_cleaned['customer_id'].str.match(r'^\d{4}-\w{4}$')]
    
    # Step 7: Additional Cleaning/Validation Steps might involve checking for logical consistency across related attributes
    # No explicit code provided here as it would depend on a deeper understanding of the specific data and its quirks

    return data_cleaned
```

## Cleaned Dataset Summary

- **Number of records:** 40 (0 record difference)
- **Number of columns:** 18
- **Missing values:** 0 (0.00%)

### First 5 rows of cleaned data

```
  customer_id  gender  age  tenure phone_service    multiple_lines internet_service online_security online_backup tech_support streaming_tv streaming_movies        contract paperless_billing             payment_method  monthly_charges  total_charges churn
0  7590-VHVEG  Female   37       1           Yes                No      Fiber optic              No            No           No           No               No  Month-to-month               Yes           Electronic check            29.85          29.85    No
1  5575-GNVDE    Male   59      34           Yes                No              DSL             Yes            No           No           No               No        One year                No               Mailed check            56.95        1889.50    No
2  3668-QPYBK    Male   41       2           Yes                No              DSL             Yes           Yes           No           No               No  Month-to-month               Yes               Mailed check            53.85         108.15   Yes
3  7795-CFOCW    Male   56      45            No  No phone service              DSL             Yes            No          Yes          Yes               No        One year                No  Bank transfer (automatic)            42.30        1840.75    No
4  9237-HQITU  Female   33       2           Yes                No      Fiber optic              No            No           No           No               No  Month-to-month               Yes           Electronic check            70.70         151.65   Yes
```

## Comparison: Original vs. Cleaned

- **Records:** 0 (0.00% change)
- **Missing values:** -3 (-100.00% change)

### Changes by Column

| Column | Original Missing | Cleaned Missing | Change |
|--------|-----------------|-----------------|--------|
| customer_id | 0 | 0 | 0 |
| gender | 0 | 0 | 0 |
| age | 0 | 0 | 0 |
| tenure | 0 | 0 | 0 |
| phone_service | 0 | 0 | 0 |
| multiple_lines | 0 | 0 | 0 |
| internet_service | 0 | 0 | 0 |
| online_security | 0 | 0 | 0 |
| online_backup | 0 | 0 | 0 |
| tech_support | 0 | 0 | 0 |
| streaming_tv | 0 | 0 | 0 |
| streaming_movies | 0 | 0 | 0 |
| contract | 0 | 0 | 0 |
| paperless_billing | 1 | 0 | -1 |
| payment_method | 0 | 0 | 0 |
| monthly_charges | 2 | 0 | -2 |
| total_charges | 0 | 0 | 0 |
| churn | 0 | 0 | 0 |


## Conclusion

The data cleaning process successfully addressed missing values, outliers, and data type issues in the churn dataset. The most significant changes were:

- 3 missing values were removed
