import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt

# Load dataset
file_path = 'Employee_Performance_Retention.csv'
df = pd.read_csv(file_path)

# Preprocessing
# Assume 'Attrition' is the target column, and drop rows with missing target
if 'Attrition' not in df.columns:
    raise ValueError('Attrition column not found in dataset.')
df = df.dropna(subset=['Attrition'])

# Encode categorical variables
df_encoded = pd.get_dummies(df.drop('Attrition', axis=1))
X = df_encoded

y = df['Attrition'].astype('category').cat.codes  # Encode target as numeric

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
report = classification_report(y_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('Classification Report:\n', report)

# Feature Importance
importances = rf.feature_importances_
features = X.columns
indices = importances.argsort()[::-1]

plt.figure(figsize=(10,6))
plt.title('Feature Importances')
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()
