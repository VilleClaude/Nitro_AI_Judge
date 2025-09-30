import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""# Load Data

"""

import csv

def read_csv_with_csv_module(file_path):
    data = []
    with open(file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            data.append(row)
    return data

train_data_list = read_csv_with_csv_module('/content/train_data_brain.csv')
test_data_list = read_csv_with_csv_module('/content/test_data_brain.csv')

train = pd.DataFrame(train_data_list[1:], columns=train_data_list[0])
test = pd.DataFrame(test_data_list[1:], columns=test_data_list[0])

print("Train DataFrame loaded successfully:")
display(train.head())
print("\nTest DataFrame loaded successfully:")
display(test.head())

train.info()

test.info()

sample = pd.read_csv('/content/sample_output_brain.csv')
sample.head()

sample.info()

"""# Subtask 1

"""

train_pixels = train['pixels'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')]).tolist()

max_len = max(len(p) for p in train_pixels)

train_padded_pixels = [p + [0] * (max_len - len(p)) for p in train_pixels]

train_pixels_array = np.array(train_padded_pixels)

global_mean_vector = np.mean(train_pixels_array, axis=0)
train_normalized_pixels = train_pixels_array - global_mean_vector


print("\nShape of normalized training pixel data:", train_normalized_pixels.shape)

train_pixels_array

"""# Subtask 2

"""

train_cleaned = train.dropna(subset=['class'])

X_train = train_normalized_pixels[train_cleaned.index]
y_train = train_cleaned['class'].astype(int)

test_pixels = test['pixels'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')]).tolist()

test_padded_pixels = [p + [0] * (max_len - len(p)) for p in test_pixels]

test_pixels_array = np.array(test_padded_pixels)
X_test = test_pixels_array - global_mean_vector

print("Shape of training features (X_train):", X_train.shape)
print("Shape of training labels (y_train):", y_train.shape)
print("Shape of test features (X_test):", X_test.shape)

from sklearn.model_selection import train_test_split

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print("Shape of training features (split):", X_train_split.shape)
print("Shape of validation features (split):", X_val_split.shape)
print("Shape of training labels (split):", y_train_split.shape)
print("Shape of validation labels (split):", y_val_split.shape)

import lightgbm as lgb
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

lgbm_model = lgb.LGBMClassifier(objective='binary',
                                 metric='binary_logloss',
                                 n_estimators=1000,
                                 learning_rate=0.05,
                                 num_leaves=31,
                                 max_depth=-1,
                                 min_child_samples=20,
                                 subsample=0.8,
                                 colsample_bytree=0.8,
                                 random_state=42,
                                 n_jobs=-1,
                                 scale_pos_weight=(y_train_split == 0).sum() / (y_train_split == 1).sum()
                                )

print("Training LightGBM model with early stopping...")
lgbm_model.fit(X_train_split, y_train_split,
               eval_set=[(X_val_split, y_val_split)],
               eval_metric='binary_logloss',
               callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

print("LightGBM model trained successfully.")

print("\nEvaluating LightGBM model on validation set...")
y_val_pred_lgbm = lgbm_model.predict(X_val_split)

accuracy_lgbm = accuracy_score(y_val_split, y_val_pred_lgbm)
f1_lgbm = f1_score(y_val_split, y_val_pred_lgbm)
precision_lgbm = precision_score(y_val_split, y_val_pred_lgbm)
recall_lgbm = recall_score(y_val_split, y_val_pred_lgbm)

print(f"Validation Accuracy: {accuracy_lgbm:.4f}")
print(f"Validation F1-Score: {f1_lgbm:.4f}")
print(f"Validation Precision: {precision_lgbm:.4f}")
print(f"Validation Recall: {recall_lgbm:.4f}")

print("\nMaking predictions on test data with LightGBM model...")
test_predictions_lgbm = lgbm_model.predict(X_test)

print("Shape of test predictions (LightGBM):", test_predictions_lgbm.shape)
print("\nFirst 10 test predictions (LightGBM):")
print(test_predictions_lgbm[:10])

test_predictions = lgbm_model.predict(X_test)

print("Shape of test predictions (LightGBM):", test_predictions.shape)
print("\nFirst 10 test predictions (LightGBM):")
print(test_predictions[:10])

df_train = train.copy()
df_test = test.copy()

subtask1_rows = []

i = 0

for id_train in df_train["id"]:
     answer_ls = train_normalized_pixels[i]
     i += 1
     answer_ls = np.array(answer_ls)
     subtask1_rows.append((1, id_train, answer_ls))


y_pred = test_predictions.flatten().tolist()

subtask2_rows = []
for id_, pred in zip(df_test["id"], y_pred):
    subtask2_rows.append((2, id_, pred))


submission_rows = subtask1_rows + subtask2_rows
df_submission = pd.DataFrame(submission_rows, columns=["subtaskID", "datapointID", "answer"])

df_submission.to_csv("submission.csv", index=False)

print("Submission file 'submission.csv' created successfully.")
display(df_submission.head())

df_submission.info()

