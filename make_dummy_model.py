import numpy as np, joblib, sklearn.linear_model

# Dummy data – 100 samples, 13 features (the UI expects 13 inputs)
X = np.random.rand(100, 13)
y = np.random.randint(0, 2, size=100)

model = sklearn.linear_model.LogisticRegression()
model.fit(X, y)

# Save with the exact name the app looks for
joblib.dump(model, "heart_disease_model.pkl")
print("Dummy model saved as heart_disease_model.pkl")