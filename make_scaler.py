import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Generate dummy data with 13 columns (the model expects 13 features)
X_dummy = np.random.rand(100, 13)   # 100 rows, 13 columns

# Fit a StandardScaler on the dummy data
scaler = StandardScaler()
scaler.fit(X_dummy)

# Save the scaler so the Streamlit app can load it
joblib.dump(scaler, "scaler.pkl")
print("scaler.pkl has been created")