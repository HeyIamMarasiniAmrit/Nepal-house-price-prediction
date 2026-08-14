
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

# Warnings & Settings Setup
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = [10, 5]


# Load dataset
full_data = pd.read_csv('/content/USA_Housing.csv')
print('Data Shape:', full_data.shape)
print(full_data.head(5))

# Clean column names
full_data.columns = full_data.columns.str.strip()

# Missing Values Heatmap
sns.heatmap(full_data.isnull(), yticklabels=False, cbar=False, cmap='tab20c_r')
plt.title('Missing Data: Training Set')
plt.show()

# Drop unnecessary columns and missing values
if 'Address' in full_data.columns:
    full_data.drop('Address', axis=1, inplace=True)

full_data.dropna(inplace=True)

print("\nData summary:")
print(full_data.describe())

# ==========================================
# 3. Features & Target Variable Selection
# ==========================================
X = full_data.drop('Price', axis=1)
y = full_data['Price']  # Fixed: removed trailing comma

print("\nFeatures:", X.columns.tolist())

# Feature Scaling
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.10, random_state=101
)


lm = LinearRegression()
lm.fit(X_train, y_train)

y_pred = lm.predict(X_test)

print("\n--- Linear Regression Performance ---")
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R2 Score: {r2 * 100:.2f}%")

# Plot Actual vs Predicted
sns.scatterplot(x=y_test, y=y_pred, color='blue', label='Actual Data points')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Line')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices (Linear Regression)')
plt.legend()
plt.show()

# Residual Analysis Plot
residual = y_test - y_pred
sns.histplot(residual, kde=True)
plt.title('Residuals Distribution')
plt.xlabel('Residual Value')
plt.show()


models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=101),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=101),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=101)
}

print("\n" + "=" * 50)
print(f"{'Model Name':<20} | {'RMSE':<15} | {'R2 Score (%)':<10}")
print("-" * 50)

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    m_rmse = np.sqrt(mean_squared_error(y_test, predictions))
    m_r2 = r2_score(y_test, predictions) * 100

    print(f"{name:<20} | {m_rmse:<15.2f} | {m_r2:<10.2f}%")
print("=" * 50)