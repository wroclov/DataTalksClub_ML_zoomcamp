import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/04-evaluation/homework.md

# Data preparation

df = pd.read_csv('course_lead_scoring_2026.csv')

print("Missing values before substitutions:")
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

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=1
)

# Extract the target column
y_train = df_train['converted']
y_val = df_val['converted']
y_test = df_test['converted']

# Remove the target from the feature datasets
df_train = df_train.drop(columns=['converted'])
df_val = df_val.drop(columns=['converted'])
df_test = df_test.drop(columns=['converted'])


# Question 1: ROC AUC feature importance
# ROC AUC could also be used to evaluate feature importance of numerical variables.

numerical_features = [
    'lead_score',
    'number_of_courses_viewed',
    'interaction_count',
    'annual_income'
]

auc_scores = {}

for feature in numerical_features:
    auc = roc_auc_score(y_train, df_train[feature])

    if auc < 0.5:
        auc = roc_auc_score(y_train, -df_train[feature])

    auc_scores[feature] = auc

    print(f'{feature}: {auc:.4f}')

print("\nAUC scores:")
print(pd.Series(auc_scores).sort_values(ascending=False))

print(f"\nFeature with the highest AUC: {max(auc_scores, key=auc_scores.get)}")