import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv('course_lead_scoring_2026.csv')

print("Missing values before imputation:")
print(df.isnull().sum())

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(
    include=["object", "category", "string"]
).columns

numerical_cols = df.select_dtypes(
    include=["number"]
).columns

# Replace missing categorical values with 'NA'
df[categorical_cols] = df[categorical_cols].fillna("NA")

# Replace missing numerical values with 0.0
df[numerical_cols] = df[numerical_cols].fillna(0.0)

# Check missing values after substitutions
print("\nMissing values after substitutions:")
print(df.isnull().sum())


# Question 1
# What is the most frequent observation (mode) for the column industry?

print(df['industry'].mode())
print(df['industry'].value_counts())
print(df['industry'].describe(include='all'))

# Question 2
# Create the correlation matrix for the numerical features of your dataset.
# In a correlation matrix, you compute the correlation coefficient between every pair of features.

# Create the correlation matrix for numerical features
numerical_features = df.select_dtypes(include='number').drop(
    columns=['converted'], errors='ignore'
)

correlation_matrix = numerical_features.corr()
print("correlation matrix:\n",correlation_matrix)

# Compare only the pairs listed in the question
pairs = [
    ('interaction_count', 'lead_score'),
    ('number_of_courses_viewed', 'lead_score'),
    ('number_of_courses_viewed', 'interaction_count'),
    ('annual_income', 'interaction_count')
]

for feature1, feature2 in pairs:
    print(
        f'{feature1} and {feature2}: '
        f'{correlation_matrix.loc[feature1, feature2]:.4f}'
    )

# Split the data using the exact requested calls
df_full_train, df_test = train_test_split(
    df, test_size=0.2, random_state=42
)

df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=42
)

# Separate the target and remove it from the feature dataframes
y_train = df_train['converted']
y_val = df_val['converted']
y_test = df_test['converted']

df_train = df_train.drop(columns=['converted'])
df_val = df_val.drop(columns=['converted'])
df_test = df_test.drop(columns=['converted'])

# Question 3
# Calculate the mutual information score between converted and other categorical variables in the dataset.
# Use the training set only.
# Round the scores to 2 decimals using round(score, 2).

# Define a function to calculate mutual information
def calculate_mi(series):
    return round(mutual_info_score(series, y_train), 2)

# Calculate scores for categorical features in the training set
categorical_columns = [
    'industry',
    'location',
    'lead_source',
    'employment_status'
]

mi_scores = df_train[categorical_columns].apply(calculate_mi)

print("mi_scores:\n",mi_scores.sort_values(ascending=False))
