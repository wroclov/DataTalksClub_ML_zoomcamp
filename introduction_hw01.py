import pandas as pd
import numpy as np

# https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/01-intro/homework.md

# Q1. Pandas version
print("pandas version:", pd.__version__)

df = pd.read_csv('car_fuel_efficiency_2026.csv')

# Q2. Records count
print(df.columns)
print(df.count)

# Q3. Fuel types
print("Unique values in fuel_type:",df['fuel_type'].unique())
print("Unique values count each column:\n",df.nunique())
print("Unique values count in fuel_type:",df['fuel_type'].nunique(),"\n\n")

# Q4. Missing values
print("N/A values in each column:\n",df.isna().sum())
print("N/A sum:",df.isna().sum().sum(),"\n\n")

# Q5. Max fuel efficiency
print("Max fuel_efficiency for Asia:", df.loc[df['origin'] == 'Asia', 'fuel_efficiency_mpg'].max())

# Q6. Median value of horsepower
# 1. Median horsepower before filling missing values
print("Horsepower_median: ", df['horsepower'].median())
print("Stats_describe:\n", df['horsepower'].describe())


# 2. Most frequent horsepower
most_frequent = df['horsepower'].mode()[0]
print("Most frequent: ", most_frequent)

# 3. Fill missing horsepower values
df['horsepower'] = df['horsepower'].fillna(most_frequent)

# 4. Median horsepower after filling missing values
print("Median_after_fill: ", df['horsepower'].median())
# stats describe
print("Stats_describe_after_fill:\n", df['horsepower'].describe())

# Q7. Sum of weights

X = df.loc[df['origin'] == 'Asia', ['vehicle_weight', 'model_year']].head(7).values
print(X)
XTX = X.T @ X
print(XTX)

XTX_inv = np.linalg.inv(XTX)

print(XTX_inv)

y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

w = XTX_inv @ X.T @ y

print("W sum:", w.sum())


# other stuff
#displaying last entries/rows
print(df.tail(7))
# with all columns visible
# alternative to set pd.set_option("display.max_columns", None)
print(df.tail(7).to_string())

# provide stats for each column which numerical
print(df.describe().round(2).to_string())

print(
    "GoupBy usage,origin showing min fuel_efficiency_mpg:\n",
    df.groupby('origin').fuel_efficiency_mpg.min().rename('min_fuel_efficiency_mpg')
)

print(
    "GoupBy usage,origin showing min fuel_efficiency_mpg:\n",
    df.groupby('origin').fuel_efficiency_mpg.min().reset_index(name='min_fuel_efficiency_mpg')
)



