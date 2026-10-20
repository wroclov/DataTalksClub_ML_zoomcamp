import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

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

# Question 2: Training the model

# One-hot encoding
train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

# Train the model
model = LogisticRegression(
    solver='liblinear',
    C=1.0,
    max_iter=1000
)
model.fit(X_train, y_train)

# Predict probabilities for the positive class
y_pred = model.predict_proba(X_val)[:, 1]

# Calculate ROC AUC
auc = roc_auc_score(y_val, y_pred)
print(f"\nQ2 AUC of the model: {round(auc, 3)}")

# AUC stands for Area Under the ROC Curve.
# The ROC curve measures how well a scoring system separates positive cases from negative cases
# as you vary the treshold to classify them.
#
# How to interpret AUC:
# 0.5 No ranking ability -> Equivalent to random ranking, on average.
# 0.7 Some ranking ability -> Converted leads tend to receive higher scores.
# 0.9 Strong ranking ability -> Converted leards are usually ranked above non-converted leads.

# AUC is the probability that a randomly selected positive example receives a higher score
# than a randomly selected negative example

# Question 3: Precision and Recall

# Precision = TruePositive / (TruePositive + FalsePositive)
# Recall = TruePositive / (TruePositive + FalseNegative)

# Predict probabilities for class 1 (converted)
# returns the estimated probability that each lead converts
y_pred_proba = model.predict_proba(X_val)[:, 1]

results = []

thresholds = np.arange(0.0, 1.01, 0.01)
#print(thresholds)

for threshold in thresholds:

    # Convert probaiblities to binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)

    # calculates the proportion of predicted conversions that actually converted
    precision = precision_score(y_val, y_pred, zero_division=0)

    # calculates the proportion of actual conversions the model successfully identified
    recall = recall_score(y_val, y_pred, zero_division=0)

    results.append({
        'threshold': round(threshold,2),
        'precision': precision,
        'recall': recall
    })

results_df = pd.DataFrame(results)
print(results_df)

plt.figure(figsize=(10, 6))

plt.plot(results_df['threshold'], results_df['precision'], label='Precision')
plt.plot(results_df['threshold'], results_df['recall'], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision and Recall vs. Threshold')
plt.legend()
plt.grid(True)

#plt.show()

results_df['difference'] = abs(
    results_df['precision'] - results_df['recall']
)

valid = results_df[
    ~((results_df['precision'] == 0) &
      (results_df['recall'] == 0))
]

intersection = valid.loc[valid['difference'].idxmin()]

print(f"\nQ3 - Precision and recall intersect at : {intersection}")

# Question 4: F1 score
# Precision and recall are conflicting - when one grows, the other goes down.
# That's why they are often combined into the F1 score - a metrics that takes into account both
# F1 = 2 * (P * R) / (P + R)

f1_scores = []

for threshold in thresholds:

    # Convert probaiblities to binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)

    f1 = f1_score(y_val, y_pred, zero_division=0)
    # calculates the proportion of predicted conversions that actually converted

    f1_scores.append(f1)

results_df['f1'] = f1_scores

# Find the threshold with the highest F1 score

best_row = results_df.loc[results_df['f1'].idxmax()]
print(best_row)

print(max(results_df['f1']))

# Question 5: 5-Fold CV
# Use the KFold class from Scikit-Learn to evaluate our model on 5 different folds

kfold = KFold(n_splits=5, shuffle=True, random_state=1)
scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train['converted']
    y_val = df_val['converted']

    train_dicts = df_train.drop(columns=['converted']).to_dict(orient='records')
    val_dicts = df_val.drop(columns=['converted']).to_dict(orient='records')

    dv = DictVectorizer(sparse=False)

    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)

    model = LogisticRegression(
        solver='liblinear',
        C=1.0,
        max_iter=1000
    )

    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print(f"AUC scores: {scores}")
print(f"Mean AUC: {np.mean(scores)}")
print(f"Standard deviation: {np.std(scores)}")






