# Data Cleaning

## About
This project cleans a Google Trends dataset using Python and Pandas.

## Dataset
- **Raw data:** `trends (1).csv` (26955 rows, 5 columns)
- **Cleaned data:** `cleaned_trends.csv` (26945 rows, 5 columns)

## Cleaning Steps Done
1. Removed 10 duplicate rows
2. Stripped extra whitespace from text columns
3. Standardized location names to title case
4. Renamed columns (location → country, rank → trend_rank, query → search_query)
5. Assigned proper data types (year and rank as int, country as category)
6. Verified final dataset - no missing values or duplicates

## Tools Used
- Python
- Pandas
- Jupyter Notebook
