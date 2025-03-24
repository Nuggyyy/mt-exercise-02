import pandas as pd
import glob

# Read all CSV files
files = glob.glob('perplexity_dropout_*.csv')
dfs = []

for file in files:
    dropout = float(file.split('_')[-1].replace('.csv', ''))
    df = pd.read_csv(file)
    df['Dropout'] = dropout
    dfs.append(df)

# Combine all data
combined_df = pd.concat(dfs)

# Create validation perplexity table
valid_table = combined_df.pivot(
    index='Epoch',
    columns='Dropout',
    values='Valid_PPL'
)
valid_table.columns = [f'Dropout {c}' for c in valid_table.columns]

# Create training perplexity table
train_table = combined_df.pivot(
    index='Epoch',
    columns='Dropout',
    values='Train_PPL'
)
train_table.columns = [f'Dropout {c}' for c in train_table.columns]

# Create test perplexity table (will only have final values)
test_table = combined_df[combined_df['Epoch'] == 'final'].pivot(
    index='Epoch',
    columns='Dropout',
    values='Test_PPL'
)
test_table.columns = [f'Dropout {c}' for c in test_table.columns]

# Save tables
valid_table.to_csv('validation_perplexities.csv')
train_table.to_csv('training_perplexities.csv')
test_table.to_csv('test_perplexities.csv')

# Print tables
print("Validation Perplexities:")
print(valid_table)
print("\nTraining Perplexities:")
print(train_table)
print("\nTest Perplexities:")
print(test_table)