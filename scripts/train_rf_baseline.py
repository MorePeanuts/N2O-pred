#!/usr/bin/env python
# coding: utf-8

# # Train Random Forest Baseline Model
# 
# Schema 3: Use Random Forest as the baseline

# In[ ]:


import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# ## 1. Load Data

# In[ ]:


data_dir = Path('../datasets')
train_df = pd.read_csv(data_dir / 'rf_data_train.csv')
val_df = pd.read_csv(data_dir / 'rf_data_val.csv')

print(f"Train samples: {len(train_df)}")
print(f"Val samples: {len(val_df)}")
print(f"Features: {train_df.columns.tolist()}")


# In[ ]:


# Separate features and target
target_col = 'Daily fluxes'
feature_cols = [col for col in train_df.columns if col != target_col]

X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_val = val_df[feature_cols]
y_val = val_df[target_col]

print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")


# In[ ]:


# Handling Categorical Features (One-Hot Encoding)
categorical_cols = ['crop_class', 'fertilization_class', 'appl_class']
X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=False)
X_val_encoded = pd.get_dummies(X_val, columns=categorical_cols, drop_first=False)

# Ensure the columns in the training set and validation set are consistent
missing_cols = set(X_train_encoded.columns) - set(X_val_encoded.columns)
for col in missing_cols:
    X_val_encoded[col] = 0

extra_cols = set(X_val_encoded.columns) - set(X_train_encoded.columns)
for col in extra_cols:
    X_train_encoded[col] = 0

X_val_encoded = X_val_encoded[X_train_encoded.columns]

print(f"After encoding - X_train shape: {X_train_encoded.shape}")
print(f"After encoding - X_val shape: {X_val_encoded.shape}")


# ## 2. Train the Random Forest Model

# In[ ]:


# Create Task Directory
task_name = f"train_rf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
output_dir = Path(f'../outputs/{task_name}')
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {output_dir.absolute()}")


# In[ ]:


# Configuration
config = {
    'model_type': 'RandomForest',
    'task_name': task_name,
    'timestamp': datetime.now().isoformat(),
    'train_samples': len(X_train_encoded),
    'val_samples': len(X_val_encoded),
    'n_features': X_train_encoded.shape[1]
}

# Save Configuration
with open(output_dir / 'config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Config saved")


# In[ ]:


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("Starting GridSearchCV...")
print(f"Parameter grid: {param_grid}")

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    rf, param_grid, cv=3, scoring='neg_mean_squared_error',
    verbose=2, n_jobs=-1
)

grid_search.fit(X_train_encoded, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score (neg_MSE): {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_


# ## 3. Evaluation Model

# In[ ]:


# Predicting
train_pred = best_model.predict(X_train_encoded)
val_pred = best_model.predict(X_val_encoded)

# Calculate Metrics
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
train_mae = mean_absolute_error(y_train, train_pred)
train_r2 = r2_score(y_train, train_pred)

val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
val_mae = mean_absolute_error(y_val, val_pred)
val_r2 = r2_score(y_val, val_pred)

print("="*60)
print("Model Evaluation")
print("="*60)
print("\nTrain Metrics:")
print(f"  RMSE: {train_rmse:.4f}")
print(f"  MAE:  {train_mae:.4f}")
print(f"  R²:   {train_r2:.4f}")
print("\nValidation Metrics:")
print(f"  RMSE: {val_rmse:.4f}")
print(f"  MAE:  {val_mae:.4f}")
print(f"  R²:   {val_r2:.4f}")
print("="*60)


# In[ ]:


# Save Metrics
metrics = {
    'train': {
        'rmse': float(train_rmse),
        'mae': float(train_mae),
        'r2': float(train_r2)
    },
    'val': {
        'rmse': float(val_rmse),
        'mae': float(val_mae),
        'r2': float(val_r2)
    },
    'best_params': grid_search.best_params_
}

with open(output_dir / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("Metrics saved")


# ## 4. Visualization

# In[ ]:


# Create Visualization Directory
fig_dir = output_dir / 'figures'
fig_dir.mkdir(exist_ok=True)

# Predicted vs Actual Scatter Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Train set
axes[0].scatter(y_train, train_pred, alpha=0.5, s=10)
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0].set_xlabel('True Daily Fluxes')
axes[0].set_ylabel('Predicted Daily Fluxes')
axes[0].set_title(f'Train Set (R²={train_r2:.3f})')
axes[0].grid(True, alpha=0.3)

# Validation set
axes[1].scatter(y_val, val_pred, alpha=0.5, s=10)
axes[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
axes[1].set_xlabel('True Daily Fluxes')
axes[1].set_ylabel('Predicted Daily Fluxes')
axes[1].set_title(f'Validation Set (R²={val_r2:.3f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / 'predictions_scatter.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Figure saved to {fig_dir}")


# In[ ]:


# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X_train_encoded.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importance.head(20))

# Save feature importance
feature_importance.to_csv(output_dir / 'feature_importance.csv', index=False)

# Visualize top 20 features
fig, ax = plt.subplots(figsize=(10, 8))
feature_importance.head(20).plot(x='feature', y='importance', kind='barh', ax=ax)
ax.set_xlabel('Importance')
ax.set_title('Top 20 Feature Importances')
plt.tight_layout()
plt.savefig(fig_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()


# ## 5. Save the Model

# In[ ]:


# Save the best model
with open(output_dir / 'best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save prediction results
predictions = {
    'train': {
        'true': y_train.values,
        'pred': train_pred
    },
    'val': {
        'true': y_val.values,
        'pred': val_pred
    }
}

with open(output_dir / 'predictions.pkl', 'wb') as f:
    pickle.dump(predictions, f)

print(f"\nModel and predictions saved to {output_dir}")
print("="*60)
print("Training completed!")
print("="*60)

