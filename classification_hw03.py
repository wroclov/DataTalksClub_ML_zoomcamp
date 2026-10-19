import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/03-classification/homework.md

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

# Questions 4-6
# Logistic regression, feature elimination, and regularization

# Convert dataframes to dictionaries
train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')

# One-hot encode categorical features
# Remember that we have several categorical variables in the dataset. Include them using one-hot encoding.
dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)


# Shared function to train and evaluate logistic regression
def train_and_evaluate(X_train, X_val, y_train, y_val, C=1.0):
    model = LogisticRegression(
        solver='liblinear',
        C=C,
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return accuracy_score(y_val, y_pred)


# Question 4
# Train logistic regression and calculate validation accuracy
# Fit the model on the training dataset.
# To make sure the results are reproducible across different versions of Scikit-Learn, fit the model with these parameters:
# model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
# Calculate the accuracy on the validation dataset and round it to 2 decimal digits.

original_accuracy = train_and_evaluate(
    X_train, X_val, y_train, y_val
)

print("Q4 - Validation accuracy:", round(original_accuracy, 2))


# Question 5
# Find the least useful feature using feature elimination
# Train a model using the same features and parameters as in Q4 (without rounding).
# Now exclude each feature from this set and train a model without it. Record the accuracy for each model.
# For each feature, calculate the difference between the original accuracy and the accuracy without the feature.

features_to_test = [
    'lead_source',
    'number_of_courses_viewed',
    'interaction_count'
]

print("\nQ5 - Feature elimination:")
print("Original accuracy:", original_accuracy)

for feature in features_to_test:
    # Remove the feature from the original dictionaries
    train_reduced = [
        {k: v for k, v in row.items() if k != feature}
        for row in train_dicts
    ]

    val_reduced = [
        {k: v for k, v in row.items() if k != feature}
        for row in val_dicts
    ]

    # Refit one-hot encoding without the removed feature
    dv_reduced = DictVectorizer(sparse=False)

    X_train_reduced = dv_reduced.fit_transform(train_reduced)
    X_val_reduced = dv_reduced.transform(val_reduced)

    accuracy_without_feature = train_and_evaluate(
        X_train_reduced, X_val_reduced, y_train, y_val
    )

    difference = original_accuracy - accuracy_without_feature

    print(
        f"{feature}: accuracy without feature = "
        f"{accuracy_without_feature:.6f}, "
        f"difference = {difference:.6f}"
    )


# Question 6
# regularized logistic regression.
# Test different regularization strengths (C)
# Let's try the following values of the parameter C: [0.000001, 0.00001, 0.0001, 0.001].
# Train models using all the features as in Q4.
# Calculate the accuracy on the validation dataset and round it to 3 decimal digits.

C_values = [0.000001, 0.00001, 0.0001, 0.001]
results = {}

print("\nQ6 - Regularization:")

for C in C_values:
    accuracy = train_and_evaluate(
        X_train, X_val, y_train, y_val, C=C
    )

    results[C] = round(accuracy, 3)
    print(f"C={C}: accuracy={results[C]:.3f}")

# Choose the best accuracy; if tied, select the smallest C
best_accuracy = max(results.values())
best_C = min(C for C, acc in results.items() if acc == best_accuracy)

print("Best C:", best_C)
print("Best accuracy:", best_accuracy)
