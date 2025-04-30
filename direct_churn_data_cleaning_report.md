# Churn Data Cleaning Report

**Generated on:** 2025-04-30 19:13:40

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

Given the information and user instructions outlined, here are the ideal steps for cleaning and preprocessing the churn dataset:

1. **Remove Duplicate Records**:
    - Start by removing duplicate customer records based on `customer_id` since every customer should have a unique record. This ensures the integrity of our analysis by eliminating redundancy.

    ```python
    dataframe.drop_duplicates(subset='customer_id', keep='first', inplace=True)
    ```

2. **Impute Missing Values for Numerical Columns**:
    - For the `monthly_charges` column, impute missing values with the median, which is more robust to outliers than the mean.

    ```python
    dataframe['monthly_charges'].fillna(dataframe['monthly_charges'].median(), inplace=True)
    ```

3. **Impute Missing Values for Categorical Columns**:
    - Fill missing categorical values in `paperless_billing` with "Unknown" to maintain data integrity without assuming any specific trend or bias.

    ```python
    dataframe['paperless_billing'].fillna('Unknown', inplace=True)
    ```

4. **Convert Data Types**:
    - Convert `total_charges` to numeric, ensuring non-numeric values are handled appropriately (e.g., set as NaN and then impute).

    ```python
    dataframe['total_charges'] = pd.to_numeric(dataframe['total_charges'], errors='coerce')
    ```

    - After conversion, impute any newly created NaNs in `total_charges` if required (it depends if the user wants to keep these records or not, but in this scenario, it wasn't a specific requirement to impute these).
   
5. **Remove Outliers in Monthly_Charges**:
    - Identify and remove outliers in `monthly_charges`, defined as values outside 3 standard deviations from the mean.

    ```python
    mean = dataframe['monthly_charges'].mean()
    std_dev = dataframe['monthly_charges'].std()
    cutoff = std_dev * 3
    dataframe = dataframe[(dataframe['monthly_charges'] <= mean + cutoff) & 
                          (dataframe['monthly_charges'] >= mean - cutoff)]
    ```

6. **Create New Categorical Column from Tenure**:
    - Bin `tenure` into categories (0-12, 13-24, 25-48, 49+ months) and create a new column `tenure_group`.

    ```python
    bins = [0, 12, 24, 48, float('inf')]
    names = ['0-12 months', '13-24 months', '25-48 months', '49+ months']
    dataframe['tenure_group'] = pd.cut(dataframe['tenure'], bins, labels=names)
    ```

7. **Convert Categorical Columns**:
    - Convert all columns with categorical data into the `category` data type to optimize memory usage and facilitate analysis.

    ```python
    categorical_columns = dataframe.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        dataframe[column] = dataframe[column].astype('category')
    ```

8. **Create a Binary Indicator for Churn**:
    - Convert the `churn` column to a binary indicator (1 = Yes, 0 = No).

    ```python
    dataframe['churn'] = dataframe['churn'].map({'Yes': 1, 'No': 0, 'No ': 0}) # including 'No ' to handle any possible whitespace issues observed in the data summary
    ```

9. **Ensure Customer IDs are Valid and Properly Formatted**:
    - Though specific criteria for validation are not provided, ensure each `customer_id` follows a predetermined format if known, or at least check for missing or irregularly formatted IDs.

    ```python
    # This is an hypothetical step as the format isn't specified
    # Example: dataframe[dataframe['customer_id'].str.match(r'^\d{4}-[A-Z]{4}$')]
    # Adjust regex based on the actual format requirement.
    ```

10. **Final Checks**:
    - Review the dataset for any remaining inconsistencies or required conversions, including a final check for any missing values or irregularities introduced by earlier steps.

By following these steps, the churn dataset will be cleaned and preprocessed effectively, adhering closely to the given instructions and ensuring a robust setup for further analysis or modeling tasks.

### Generated Cleaning Function

```python
# ----------------------------------------------------------------------
# Code generated by AI agent: data_cleaning_agent
# ----------------------------------------------------------------------

def data_cleaner(data_raw):
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer

    # Step 1: Remove duplicate records
    data_cleaned = data_raw.drop_duplicates(subset='customer_id', keep='first', inplace=False)
    
    # Step 2: Impute missing values for numerical columns (monthly_charges)
    imputer = SimpleImputer(strategy='median')
    data_cleaned['monthly_charges'] = imputer.fit_transform(data_cleaned[['monthly_charges']]).ravel()
    
    # Step 3: Impute missing values for categorical columns (paperless_billing)
    data_cleaned['paperless_billing'].fillna('Unknown', inplace=True)
    
    # Step 4: Convert data types
    data_cleaned['total_charges'] = pd.to_numeric(data_cleaned['total_charges'], errors='coerce')
    
    # Note: Depending on the use case, here you might want to impute the 'total_charges' missing values after conversion
    # For this demo, we'll proceed without explicit imputation as it wasn't a specific requirement.
    
    # Step 5: Remove outliers in Monthly_Charges
    mean = data_cleaned['monthly_charges'].mean()
    std_dev = data_cleaned['monthly_charges'].std()
    cutoff = std_dev * 3
    data_cleaned = data_cleaned[(data_cleaned['monthly_charges'] <= mean + cutoff) &
                                (data_cleaned['monthly_charges'] >= mean - cutoff)]
    
    # Step 6: Create new categorical column from Tenure
    bins = [0, 12, 24, 48, float('inf')]
    names = ['0-12 months', '13-24 months', '25-48 months', '49+ months']
    data_cleaned['tenure_group'] = pd.cut(data_cleaned['tenure'], bins, labels=names)
    
    # Step 7: Convert Categorical Columns
    categorical_columns = data_cleaned.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        data_cleaned[column] = data_cleaned[column].astype('category')
    
    # Step 8: Convert 'churn' column to a binary indicator
    data_cleaned['churn'] = data_cleaned['churn'].map({'Yes': 1, 'No': 0, 'No ': 0})
    
    # Step 9: Validate and format customer IDs (hypothetical step, adjust based on requirements)
    # Example: Assume Customer_ID should have a format like 0000-XXXX
    # data_cleaned = data_cleaned[data_cleaned['customer_id'].str.match(r'^\d{4}-[A-Z]{4}$')]
    
    # Step 10: Final Checks
    # Ensure there are no remaining missing values or inconsistencies introduced
    # This could include reviewing the data types again, ensuring no unexpected NaN values, etc.
    # Note: Specific checks or adjustments would depend on further analysis or insights from the data cleaning thus far
    
    return data_cleaned
```

## Cleaned Dataset Summary

- **Number of records:** 40 (0 record difference)
- **Number of columns:** 19
- **Missing values:** 0 (0.00%)

### First 5 rows of cleaned data

```
  customer_id  gender  age  tenure phone_service    multiple_lines internet_service online_security online_backup tech_support streaming_tv streaming_movies        contract paperless_billing             payment_method  monthly_charges  total_charges  churn  tenure_group
0  7590-VHVEG  Female   37       1           Yes                No      Fiber optic              No            No           No           No               No  Month-to-month               Yes           Electronic check            29.85          29.85      0   0-12 months
1  5575-GNVDE    Male   59      34           Yes                No              DSL             Yes            No           No           No               No        One year                No               Mailed check            56.95        1889.50      0  25-48 months
2  3668-QPYBK    Male   41       2           Yes                No              DSL             Yes           Yes           No           No               No  Month-to-month               Yes               Mailed check            53.85         108.15      1   0-12 months
3  7795-CFOCW    Male   56      45            No  No phone service              DSL             Yes            No          Yes          Yes               No        One year                No  Bank transfer (automatic)            42.30        1840.75      0  25-48 months
4  9237-HQITU  Female   33       2           Yes                No      Fiber optic              No            No           No           No               No  Month-to-month               Yes           Electronic check            70.70         151.65      1   0-12 months
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
| tenure_group | Column added | 0 | N/A |


## Conclusion

The data cleaning process successfully addressed the following issues in the churn dataset:

- 3 missing values were removed
- Column 'paperless_billing': 1 missing values were filled
- Column 'monthly_charges': 2 missing values were filled
- 1 columns were added: tenure_group
- Data types were changed for: churn (from object to int64)
