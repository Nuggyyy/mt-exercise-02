import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

# Load data
train_df = pd.read_csv('logs/train.csv')
valid_df = pd.read_csv('logs/valid.csv')

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Define dropout values for legend
dropout_values = ['0.0', '0.25', '0.5', '0.75', '1.0']
colors = sns.color_palette("viridis", len(dropout_values))

# Plot training perplexity
for i, dropout in enumerate(dropout_values):
    axes[0].plot(train_df['Epoch'], train_df[f'Dropout {dropout}'], 
                label=f'Dropout = {dropout}', color=colors[i], linewidth=2)
    
axes[0].set_title('Training Perplexity vs Epochs', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=13)
axes[0].set_ylabel('Perplexity', fontsize=13)
axes[0].legend()
axes[0].grid(True)

# Plot validation perplexity
for i, dropout in enumerate(dropout_values):
    axes[1].plot(valid_df['Epoch'], valid_df[f'Dropout {dropout}'], 
                label=f'Dropout = {dropout}', color=colors[i], linewidth=2)
    
axes[1].set_title('Validation Perplexity vs Epochs', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=13)
axes[1].set_ylabel('Perplexity', fontsize=13)
axes[1].legend()
axes[1].grid(True)

# Add text explaining what perplexity is
fig.text(0.5, 0.01, 
         "Perplexity is a measure of how well a probability model predicts a sample.\n"
         "Lower perplexity values indicate better performance.", 
         ha='center', fontsize=12, fontstyle='italic')

# Adjust layout and save
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig('logs/perplexity_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Create tables
print("\nTraining Perplexity Table:")
print(train_df.to_string(index=False))

print("\nValidation Perplexity Table:")
print(valid_df.to_string(index=False))

# Calculate the final perplexity values for each dropout rate
final_epoch = train_df['Epoch'].max()
final_train = train_df[train_df['Epoch'] == final_epoch].iloc[0, 1:].tolist()
final_valid = valid_df[valid_df['Epoch'] == final_epoch].iloc[0, 1:].tolist()

print("\nFinal Perplexity Values (Epoch 40):")
comparison_df = pd.DataFrame({
    'Dropout Rate': dropout_values,
    'Training Perplexity': final_train,
    'Validation Perplexity': final_valid
})
print(comparison_df.to_string(index=False))