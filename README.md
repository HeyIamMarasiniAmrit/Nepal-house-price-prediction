# 🏠 Nepal House Price Prediction

This project predicts house prices in Nepal using Machine Learning regression models.

## 📌 Project Overview

The goal of this project is to build and compare multiple regression models to predict house prices based on various real estate features commonly found in the Nepali housing market.

### Models Used:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

## 🛠️ Tech Stack

- **Python**
- **Pandas** & **NumPy** – Data manipulation
- **Matplotlib** & **Seaborn** – Data visualization
- **Scikit-learn** – Machine Learning models and preprocessing

## 📂 Dataset

The dataset contains information about houses in Nepal with features such as:

- Location
- Land Area (Aana)
- Built-up Area
- Number of Bedrooms
- Number of Bathrooms
- Number of Floors
- Road Access
- Facing Direction
- Amenities
- Price (Target Variable)

## 🔄 Project Workflow

1. **Data Loading & Exploration**
2. **Data Cleaning**
   - Handling missing values
   - Removing unnecessary columns
3. **Feature Preprocessing**
   - Encoding categorical variables
   - Feature scaling using `StandardScaler`
4. **Train-Test Split**
5. **Model Training & Evaluation**
6. **Model Comparison**

## 📊 Evaluation Metrics

- Root Mean Squared Error (RMSE)
- R² Score

## 📈 Key Insights

- Location and Road Access are among the most important factors affecting house prices.
- Ensemble models (Random Forest and Gradient Boosting) generally perform better than simple Linear Regression.

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/your-username/nepal-house-price-prediction.git
