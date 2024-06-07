# miR-133a Expression Prediction with CatBoost

This repository contains the code used in our research paper for predicting miR-133a expression levels using CatBoost regression models. The repository includes data preprocessing, model training, evaluation, and visualization scripts.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
  - [Model Tuning](#model-tuning)
  - [Results Visualization](#results-visualization)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to predict miR-133a expression levels based on various biological and clinical features using CatBoost regression. The study includes initial model training, parameter tuning, and detailed visualizations to assess model performance.

## Dataset

The dataset used in this study consists of the following features:

- Group (Categorical)
- miR-133a Expression (Fold Change)
- Body Weight (g)
- Blood Glucose (mg/dL)
- Insulin (µIU/mL)
- HOMA-IR
- HbA1c (mmol/l)
- Total Cholesterol (mg/dL)
- HDL (mg/dL)
- LDL (mg/dL)
- Triglycerides (mg/dL)

## Installation

To run the code in this repository, you need to have Python 3.x installed along with the following packages:

```bash
pip install pandas scikit-learn catboost matplotlib
```

## Usage

### Data Preprocessing

The data preprocessing script handles the conversion of categorical variables to numerical values and splits the dataset into training and testing sets.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = {
    # (dataset details)
}

df = pd.DataFrame(data)

# Convert categorical groups to numerical values using one-hot encoding
df = pd.get_dummies(df, columns=['Group'], drop_first=True)

# Choose the target variable
target = 'miR-133a Expression (Fold Change)'

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1), df[target], test_size=0.2, random_state=42)
```

### Model Training

The initial CatBoost regression model is trained using the following script:

```python
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create a CatBoostRegressor model
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42
)

# Train the model
model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
```

### Model Tuning

The tuned CatBoost regression model with optimized parameters:

```python
# Tuned CatBoostRegressor model
tuned_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=5,
    l2_leaf_reg=8,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42
)

# Train the tuned model
tuned_model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100)

# Evaluate the tuned model
y_pred_tuned = tuned_model.predict(X_test)
mse_tuned = mean_squared_error(y_test, y_pred_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)

print(f"Tuned Mean Squared Error: {mse_tuned}")
print(f"Tuned R-squared: {r2_tuned}")
```

### Results Visualization

Visualizing the feature importance and model predictions:

```python
import matplotlib.pyplot as plt
import pandas as pd

# Combine predictions and actual values
train_data = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred})
test_data = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
combined_data = pd.concat([train_data, test_data], axis=0)

# Calculate the difference between actual and predicted values
combined_data['Difference'] = combined_data['Actual'] - combined_data['Predicted']

# Create the plot
plt.figure(figsize=(12, 10))
plt.scatter(combined_data['Actual'], combined_data['Predicted'], 
            color='#007bff', s=1000, alpha=0.8, label='Predictions',
            marker='o', edgecolors='black', linewidths=1.5)
plt.plot([combined_data['Actual'].min(), combined_data['Actual'].max()],
         [combined_data['Actual'].min(), combined_data['Actual'].max()], 
         color='#ff0000', lw=3, linestyle='--', label='Ideal Line')

# Add annotations
for i, row in combined_data.iterrows():
    plt.annotate(f"{row['Difference']:.2f}", 
                 xy=(row['Actual'], row['Predicted']), 
                 fontsize=14, ha='center', va='center',
                 color='white',
                 bbox=dict(facecolor='#007bff', edgecolor='black', boxstyle='circle,pad=0.5'))

# Customize the plot
plt.xlabel('Actual Values', fontsize=18, fontweight='bold')
plt.ylabel('Predicted Values', fontsize=18, fontweight='bold')
plt.grid(True, alpha=0.5)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()
```

## Results

### Table 2: Model Performance Metrics

| Model | Mean Squared Error | R-squared |
|-------|--------------------:|----------:|
| Initial CatBoost Model | 0.0008111481371457363 | 0.6755407451417053 |
| Tuned CatBoost Model   | 5.90116284577064e-07  | 0.9997639534861692 |



## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

