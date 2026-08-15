# nepal-house-price-prediction


import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


warnings.filterwarnings('ignore')
warnings.simplefilter(action="ignore", category=FutureWarning)
plt.rcParams['figure.figsize'] = [10, 5]
sns.set_theme(style="whitegrid")


# Load dataset
full_data = pd.read_csv('/content/archive.zip')
print(f"Initial Data Shape: {full_data.shape}\n")

# Display initial info
full_data.info()
print("\nFirst 5 Rows:")
print(full_data.head())

# Visualizing missing data
plt.figure()
sns.heatmap(full_data.isnull(), yticklabels=False, cbar=False, cmap='tab20c_r')
plt.title('Missing Data Visualization')
plt.show()

# Clean column names and drop unnecessary columns/rows
full_data.columns = full_data.columns.str.strip()

if 'Address' in full_data.columns:
    full_data.drop('Address', axis=1, inplace=True)

full_data.dropna(inplace=True)

print(f"\nData Shape after Cleaning: {full_data.shape}")
print(full_data.describe())


# Split Features (X) and Target (y)
X = full_data.drop('Price', axis=1)
y = full_data['Price']

# Select numerical features for scaling
X_numerical = X.select_dtypes(include=['number'])

# Feature Scaling
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X_numerical)

print(f"\nScaled Features Shape: {X_scaled.shape}")
print(f"Target Shape: {y.shape}")

# Train-Test Split (90% Train, 10% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.10, random_state=101
)


lm = LinearRegression()
lm.fit(X_train, y_train)

y_pred = lm.predict(X_test)

# Plotting Actual vs Predicted Prices
plt.figure()
sns.scatterplot(x=y_test, y=y_pred, color='blue', label='Actual Data points')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Line')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Linear Regression: Actual vs Predicted Prices')
plt.legend()
plt.show()

# Residual Plot
residual = y_test - y_pred
plt.figure()
sns.histplot(residual, kde=True, color='purple')
plt.title('Residuals Distribution')
plt.xlabel('Residuals (Actual - Predicted)')
plt.show()


models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=101),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=101),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=101)
}

print("\n" + "=" * 55)
print(f"{'Model Name':<20} | {'RMSE':<15} | {'R2 Score (%)':<10}")
print("=" * 55)

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions) * 100

    print(f"{name:<20} | {rmse:<15.2f} | {r2:<10.2f}%")
print("=" * 55)
