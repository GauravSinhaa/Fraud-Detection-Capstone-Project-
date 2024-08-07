import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Set random seed for reproducibility
np.random.seed(42)

# Create synthetic data
n_samples = 1000
data = {
    'transaction_id': np.arange(n_samples),
    'transaction_amount': np.random.uniform(10, 1000, n_samples),
    'transaction_location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
    'transaction_time': np.random.randint(0, 24, n_samples),
    'is_fraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
}

df = pd.DataFrame(data)

# Display the first few rows of the dataset
df.head(10)


# Define features and target variable
X = df.drop(columns='is_fraud')
y = df['is_fraud']

# Preprocess categorical and numerical data
numeric_features = ['transaction_amount', 'transaction_time']
categorical_features = ['transaction_location']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit and transform the data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


# Initialize and train the model
model = GaussianNB()
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

# Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


print("Conclusion:")
print(f"The model achieved an accuracy of {accuracy:.2f}.")
print("Next steps could include tuning the model parameters, exploring other models, and using a more complex dataset.")
