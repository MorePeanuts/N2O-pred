#!/usr/bin/env python
# coding: utf-8

# # Training RNN Model – Observation Step Schemes
# 
# Scheme 1: RNN with observation time as the step, using the time interval as an input feature

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

from n2o_pred.models import N2OPredictorRNN, count_parameters
from n2o_pred.data import create_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

print(f'Random seed: {RANDOM_SEED}')


# ## 1. Load Data

# In[ ]:


data_dir = Path('../datasets/processed')

with open(data_dir / 'sequences_obs_step_train_processed.pkl', 'rb') as f:
    train_sequences = pickle.load(f)
with open(data_dir / 'sequences_obs_step_val_processed.pkl', 'rb') as f:
    val_sequences = pickle.load(f)

print(f'Train: {len(train_sequences)}, Val: {len(val_sequences)}')


# ## 2. Model Initialization and Training

# In[ ]:


BATCH_SIZE = 16
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
RNN_TYPE = 'LSTM'
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
PATIENCE = 15

train_loader = create_dataloader(
    train_sequences, 'obs_step', BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = create_dataloader(val_sequences, 'obs_step', BATCH_SIZE, shuffle=False, num_workers=0)

static_numeric_dim = train_sequences[0]['static_numeric'].shape[0]
static_categorical_dim = train_sequences[0]['static_categorical_encoded'].shape[0]
dynamic_numeric_dim = train_sequences[0]['dynamic_numeric'].shape[1]
fertilization_numeric_dim = train_sequences[0]['fertilization_numeric'].shape[1]
fertilization_categorical_dim = sum(
    train_sequences[0]['fertilization_categorical_encoded'][key].shape[1]
    for key in train_sequences[0]['fertilization_categorical_encoded']
)

model = N2OPredictorRNN(
    static_numeric_dim=static_numeric_dim,
    static_categorical_dim=static_categorical_dim,
    dynamic_numeric_dim=dynamic_numeric_dim,
    fertilization_numeric_dim=fertilization_numeric_dim,
    fertilization_categorical_dim=fertilization_categorical_dim,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    rnn_type=RNN_TYPE,
).to(device)

print(f'Model parameters: {count_parameters(model):,}')

task_name = f'rnn_obs_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
output_dir = Path(f'../outputs/{task_name}')
output_dir.mkdir(parents=True, exist_ok=True)


config = {
    'model_type': 'N2ORNN_ObsStep',
    'task_name': task_name,
    'random_seed': RANDOM_SEED,
    'hyperparameters': {
        'batch_size': BATCH_SIZE,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
        'rnn_type': RNN_TYPE,
        'learning_rate': LEARNING_RATE,
        'num_epochs': NUM_EPOCHS,
        'patience': PATIENCE,
    },
}
with open(output_dir / 'config.json', 'w') as f:
    json.dump(config, f, indent=2)

criterion = nn.MSELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

print(f'Output: {output_dir}')


# ## 3. Training Function

# In[ ]:


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, total_samples = 0, 0
    for batch in tqdm(dataloader, desc='Train', leave=False):
        static_feat = batch['static_features'].to(device)
        dynamic_feat = batch['dynamic_features'].to(device)
        targets = batch['targets'].to(device)
        lengths = batch['lengths']
        masks = batch['masks'].to(device)

        optimizer.zero_grad()
        predictions = model(static_feat, dynamic_feat, lengths)
        loss = criterion(predictions, targets)
        masked_loss = (loss * masks.float()).sum() / masks.sum()
        masked_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += masked_loss.item() * masks.sum().item()
        total_samples += masks.sum().item()
    return total_loss / total_samples


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_samples = 0, 0
    all_preds, all_targets, all_targets_orig = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Eval', leave=False):
            static_feat = batch['static_features'].to(device)
            dynamic_feat = batch['dynamic_features'].to(device)
            targets = batch['targets'].to(device)
            targets_orig = batch['targets_original']
            lengths = batch['lengths']
            masks = batch['masks'].to(device)

            predictions = model(static_feat, dynamic_feat, lengths)
            loss = criterion(predictions, targets)
            masked_loss = (loss * masks.float()).sum() / masks.sum()

            total_loss += masked_loss.item() * masks.sum().item()
            total_samples += masks.sum().item()

            for i in range(len(lengths)):
                seq_len = lengths[i].item()
                all_preds.append(predictions[i, :seq_len].cpu().numpy())
                all_targets.append(targets[i, :seq_len].cpu().numpy())
                all_targets_orig.append(targets_orig[i, :seq_len].numpy())

    avg_loss = total_loss / total_samples
    all_preds_flat = np.concatenate(all_preds)
    all_targets_flat = np.concatenate(all_targets)
    rmse = np.sqrt(np.mean((all_preds_flat - all_targets_flat) ** 2))
    mae = np.mean(np.abs(all_preds_flat - all_targets_flat))

    # Calculate R² (coefficient of determination)
    ss_res = np.sum((all_targets_flat - all_preds_flat) ** 2)
    ss_tot = np.sum((all_targets_flat - np.mean(all_targets_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return avg_loss, rmse, mae, r2, all_preds, all_targets, all_targets_orig


print('Functions defined')


# ## 4. Training Loop

# In[ ]:


history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_mae': [], 'val_r2': [], 'lr': []}
best_val_loss = float('inf')
patience_counter = 0

print('Starting training...')
for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_rmse, val_mae, val_r2, _, _, _ = evaluate(model, val_loader, criterion, device)
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_rmse'].append(val_rmse)
    history['val_mae'].append(val_mae)
    history['val_r2'].append(val_r2)
    history['lr'].append(current_lr)

    print(
        f'Epoch {epoch + 1}: TrLoss={train_loss:.6f}, ValLoss={val_loss:.6f}, RMSE={val_rmse:.6f}, MAE={val_mae:.6f}, R²={val_r2:.6f}'
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'val_r2': val_r2,
            },
            output_dir / 'best_model.pth',
        )
        print('  ✓ Best model saved')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f'Early stopping at epoch {epoch + 1}')
            break

print(f'Training completed! Best val loss: {best_val_loss:.6f}')

pd.DataFrame(history).to_csv(output_dir / 'training_log.csv', index=False)
print('Training log saved')


# ## 5. Visualization and Final Evaluation

# In[ ]:


fig_dir = output_dir / 'figures'
fig_dir.mkdir(exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes[0, 0].plot(history['train_loss'], label='Train')
axes[0, 0].plot(history['val_loss'], label='Val')
axes[0, 0].set_title('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 1].plot(history['val_rmse'])
axes[0, 1].set_title('Val RMSE')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 2].plot(history['val_mae'])
axes[0, 2].set_title('Val MAE')
axes[0, 2].grid(True, alpha=0.3)
axes[1, 0].plot(history['val_r2'])
axes[1, 0].set_title('Val R²')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 1].plot(history['lr'])
axes[1, 1].set_title('Learning Rate')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 2].axis('off')  # Hide the last subplot
plt.tight_layout()
plt.savefig(fig_dir / 'training_curves.png', dpi=150)
plt.show()

# Load the best model and evaluate
checkpoint = torch.load(output_dir / 'best_model.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

train_loss, train_rmse, train_mae, train_r2, train_preds, train_targets, train_targets_orig = (
    evaluate(model, train_loader, criterion, device)
)
val_loss, val_rmse, val_mae, val_r2, val_preds, val_targets, val_targets_orig = evaluate(
    model, val_loader, criterion, device
)

print(
    f'Final - Train Loss: {train_loss:.6f}, RMSE: {train_rmse:.6f}, MAE: {train_mae:.6f}, R²: {train_r2:.6f}'
)
print(
    f'Final - Val Loss: {val_loss:.6f}, RMSE: {val_rmse:.6f}, MAE: {val_mae:.6f}, R²: {val_r2:.6f}'
)

# Save prediction results
predictions = {
    'train': {
        'predictions': train_preds,
        'targets': train_targets,
        'targets_original': train_targets_orig,
    },
    'val': {'predictions': val_preds, 'targets': val_targets, 'targets_original': val_targets_orig},
    'metrics': {
        'train': {
            'loss': float(train_loss),
            'rmse': float(train_rmse),
            'mae': float(train_mae),
            'r2': float(train_r2),
        },
        'val': {
            'loss': float(val_loss),
            'rmse': float(val_rmse),
            'mae': float(val_mae),
            'r2': float(val_r2),
        },
    },
}
with open(output_dir / 'predictions.pkl', 'wb') as f:
    pickle.dump(predictions, f)

print(f'\n{"=" * 60}')
print(f'All results saved to: {output_dir}')
print(f'{"=" * 60}')

