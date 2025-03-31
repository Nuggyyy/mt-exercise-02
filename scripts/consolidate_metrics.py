#!/usr/bin/env python3
import os
import csv
import argparse
import glob
import pandas as pd

def consolidate_metrics(metrics_dir, output_dir):
    """
    Consolidate individual metrics files into combined CSV files with dropouts as columns.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output files
    train_output = os.path.join(output_dir, 'train.csv')
    valid_output = os.path.join(output_dir, 'valid.csv')
    test_output = os.path.join(output_dir, 'test.csv')
    
    # Get all train metrics files
    train_files = glob.glob(os.path.join(metrics_dir, 'train_dropout_*.csv'))
    valid_files = glob.glob(os.path.join(metrics_dir, 'valid_dropout_*.csv'))
    test_files = glob.glob(os.path.join(metrics_dir, 'test_dropout_*.csv'))
    
    # Process training metrics
    train_dfs = []
    for file in sorted(train_files):
        dropout = float(file.split('_')[-1].replace('.csv', ''))
        df = pd.read_csv(file)
        df.rename(columns={'Perplexity': f'Dropout {dropout}'}, inplace=True)
        train_dfs.append(df)
    
    if train_dfs:
        # Merge all dataframes on Epoch
        result = train_dfs[0]
        for df in train_dfs[1:]:
            result = pd.merge(result, df, on='Epoch', how='outer')
        
        # Write to output file
        result.to_csv(train_output, index=False)
        print(f"Training metrics consolidated to {train_output}")
    
    # Process validation metrics
    valid_dfs = []
    for file in sorted(valid_files):
        dropout = float(file.split('_')[-1].replace('.csv', ''))
        df = pd.read_csv(file)
        df.rename(columns={'Perplexity': f'Dropout {dropout}'}, inplace=True)
        valid_dfs.append(df)
    
    if valid_dfs:
        # Merge all dataframes on Epoch
        result = valid_dfs[0]
        for df in valid_dfs[1:]:
            result = pd.merge(result, df, on='Epoch', how='outer')
        
        # Write to output file
        result.to_csv(valid_output, index=False)
        print(f"Validation metrics consolidated to {valid_output}")
    
    # Process test metrics (different format)
    test_data = {}
    dropouts = []
    
    for file in sorted(test_files):
        dropout = float(file.split('_')[-1].replace('.csv', ''))
        dropouts.append(dropout)
        df = pd.read_csv(file)
        test_data[f'Dropout {dropout}'] = df.iloc[0]['Perplexity']
    
    if dropouts:
        # Create DataFrame with one row, where each dropout value is a column
        test_df = pd.DataFrame([test_data])
        test_df.to_csv(test_output, index=False)
        print(f"Test metrics consolidated to {test_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Consolidate metrics files into combined CSVs')
    parser.add_argument('--metrics-dir', type=str, default='./metrics',
                        help='directory containing metrics files')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='directory to save consolidated CSV files')
    args = parser.parse_args()
    
    consolidate_metrics(args.metrics_dir, args.output_dir)