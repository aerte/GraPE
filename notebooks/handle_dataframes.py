import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------
# Step 1: Load the DataFrames
# ---------------------------

#df1 = pd.read_csv("env/predictions_groupgat_decoupled_multitask.csv")
df1 = pd.read_csv("env/predictions_AGC_decoupled_single-task.csv")
df2 = pd.read_excel("env/icp_results_all (1).xlsx")
breakpoint()
# ---------------------------
# Step 2: Rename 'bin' to 'bin_pred'
# ---------------------------

df1_renamed = df1.rename(columns={'bin': 'bin_pred'})

# ---------------------------
# Step 3: Identify and Remove Non-common SMILES
# ---------------------------

# Identify unique SMILES in both DataFrames
smiles_df1 = set(df1_renamed['SMILES'].unique())
smiles_df2 = set(df2['SMILES'].unique())

# Determine common SMILES
common_smiles = smiles_df1.intersection(smiles_df2)
num_common = len(common_smiles)
num_df1 = len(smiles_df1)
num_df2 = len(smiles_df2)

print(f"Number of unique SMILES in df1: {num_df1}")
print(f"Number of unique SMILES in df2: {num_df2}")
print(f"Number of common unique SMILES: {num_common}")

# Identify SMILES to remove from df1
smiles_to_remove = smiles_df1 - smiles_df2
num_to_remove = len(smiles_to_remove)

print(f"Number of SMILES to remove from df1: {num_to_remove}")
print(f"SMILES to remove: {smiles_to_remove}")

# Filter df1 to retain only common SMILES
df1_filtered = df1_renamed[df1_renamed['SMILES'].isin(common_smiles)].copy()

print(f"Number of unique SMILES in df1 after filtering: {df1_filtered['SMILES'].nunique()}")

# ---------------------------
# Step 4: Select Relevant Columns from Filtered df1
# ---------------------------

df1_selected = df1_filtered[['SMILES', 
                              'Predicted_A0', 
                              'Predicted_B0', 
                              'Predicted_C0', 
                              'Predicted_D0', 
                              'Predicted_E0', 
                              'bin_pred']]

# ---------------------------
# Step 5: Merge df2 with df1_selected on 'SMILES'
# ---------------------------

merged_df = df2.merge(df1_selected, on='SMILES', how='left')

# ---------------------------
# Step 6: Create the final DataFrame with Desired Columns and Make an Explicit Copy
# ---------------------------

final_df = merged_df[['SMILES',
                      'target', 
                      'T', 
                      'Predicted_A0', 
                      'Predicted_B0', 
                      'Predicted_C0', 
                      'Predicted_D0', 
                      'Predicted_E0', 
                      'bin_pred']].copy()

def plot_distributions(df, output_path=None):
    """
    Plots the distribution of Temperature (T) and Target from the provided DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing 'T' and 'target' columns.
    - output_path (str, optional): If provided, the plot will be saved to this path.

    Returns:
    - None
    """
    # Check if required columns exist
    if not {'T', 'target'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'T' and 'target' columns.")
    
    # Set the aesthetic style of the plots
    sns.set(style="whitegrid")

    # Create a figure with two subplots side by side
    plt.figure(figsize=(14, 6))

    # **Temperature Distribution**
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    sns.histplot(df['T'], kde=True, color='skyblue', bins=30)
    plt.title('Temperature Distribution')
    plt.xlabel('Temperature (K) ')
    plt.ylabel('Frequency')

    # **Target Distribution**
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    sns.histplot(df['target'], kde=True, color='salmon', bins=30)
    plt.title('heat Capacity Distribution')
    plt.xlabel('Heat capacity (kJ/kmol⋅K)')
    plt.ylabel('Frequency')

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the plot if an output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Distribution plots saved to {output_path}")

    # Display the plots
    plt.show()

    # Optional: Create and display a joint plot
    sns.jointplot(data=df, x='T', y='target', kind='reg', color='green', height=8)
    plt.suptitle('Joint Distribution of Temperature and Target with Regression', y=1.02)
    plt.show()

plot_distributions(final_df)


# Verify the result
print("\nFinal DataFrame shape:", final_df.shape)  # Expected: (23184, 9)
print("First few rows of the final DataFrame:")
print(final_df.head())

# ---------------------------
# Step 7: Define the Cp Function with Numerical Stability
# ---------------------------

def Cp(A, B, C, D, E, T):
    epsilon = 1e-6
    T = torch.clamp(T, min=epsilon)  # Prevent division by zero
    
    ratio_C = C / T
    ratio_E = E / T
    
    # Clamp ratios to avoid numerical issues in hyperbolic functions
    ratio_C = torch.clamp(ratio_C, min=-20, max=20)
    ratio_E = torch.clamp(ratio_E, min=-20, max=20)
    
    term1 = A
    term2 = B * ((ratio_C) / torch.sinh(ratio_C)) ** 2
    term3 = D * ((ratio_E) / torch.cosh(ratio_E)) ** 2
    
    return 4*(term1 + term2 + term3)

# ---------------------------
# Step 8: Compute predicted_Cp Using Vectorized Operations
# ---------------------------

# Extract the necessary columns as torch tensors
A = torch.tensor(final_df['Predicted_A0'].values, dtype=torch.float32)
B = torch.tensor(final_df['Predicted_B0'].values, dtype=torch.float32)
C = torch.tensor(final_df['Predicted_C0'].values, dtype=torch.float32)
D = torch.tensor(final_df['Predicted_D0'].values, dtype=torch.float32)
E = torch.tensor(final_df['Predicted_E0'].values, dtype=torch.float32)
T = torch.tensor(final_df['T'].values, dtype=torch.float32)

# Compute predicted_Cp using the Cp function
predicted_Cp = Cp(A, B, C, D, E, T)

# Assign the computed values to a new column in final_df using .loc to avoid warnings
final_df.loc[:, 'predicted_Cp'] = predicted_Cp.numpy()

# Verify that no missing values were introduced
missing_pred_cp = final_df['predicted_Cp'].isnull().sum()
print(f"\nMissing values in 'predicted_Cp': {missing_pred_cp}")

# ---------------------------
# Step 9: Split the DataFrame into Train, Val, and Test Sets
# ---------------------------

# Split the DataFrame based on 'bin_pred' column
train_df = final_df[final_df['bin_pred'] == 'train'].copy()
val_df = final_df[final_df['bin_pred'] == 'val'].copy()
test_df = final_df[final_df['bin_pred'] == 'test'].copy()

# Verify the splits
print("\nDataset Splits:")
print(f"Training Set: {len(train_df)} samples")
print(f"Validation Set: {len(val_df)} samples")
print(f"Testing Set: {len(test_df)} samples")

# ---------------------------
# Step 10: Compute Performance Metrics for Each Split
# ---------------------------

# Function to compute MARE
def compute_mare(target, predicted, epsilon=1e-8):
    return (np.abs(target - predicted) / (np.abs(target) + epsilon)).mean()

# Compute metrics for Training Set
r2_train = r2_score(train_df['target'], train_df['predicted_Cp'])
mae_train = mean_absolute_error(train_df['target'], train_df['predicted_Cp'])
mare_train = compute_mare(train_df['target'], train_df['predicted_Cp'])

# Compute metrics for Validation Set
r2_val = r2_score(val_df['target'], val_df['predicted_Cp'])
mae_val = mean_absolute_error(val_df['target'], val_df['predicted_Cp'])
mare_val = compute_mare(val_df['target'], val_df['predicted_Cp'])

# Compute metrics for Testing Set
r2_test = r2_score(test_df['target'], test_df['predicted_Cp'])
mae_test = mean_absolute_error(test_df['target'], test_df['predicted_Cp'])
mare_test = compute_mare(test_df['target'], test_df['predicted_Cp'])

# Print the metrics
print("\nPerformance Metrics:")
print("Training Set:")
print(f"  R²: {r2_train:.4f}")
print(f"  MAE: {mae_train:.4f}")
print(f"  MARE: {mare_train:.4f}")

print("\nValidation Set:")
print(f"  R²: {r2_val:.4f}")
print(f"  MAE: {mae_val:.4f}")
print(f"  MARE: {mare_val:.4f}")

print("\nTesting Set:")
print(f"  R²: {r2_test:.4f}")
print(f"  MAE: {mae_test:.4f}")
print(f"  MARE: {mare_test:.4f}")

# Compute metrics for the entire dataset
r2_overall = r2_score(final_df['target'], final_df['predicted_Cp'])
mae_overall = mean_absolute_error(final_df['target'], final_df['predicted_Cp'])
mare_overall = compute_mare(final_df['target'], final_df['predicted_Cp'])

# Print the overall metrics
print("\nOverall Performance Metrics:")
print(f"  R²: {r2_overall:.4f}")
print(f"  MAE: {mae_overall:.4f}")
print(f"  MARE: {mare_overall:.4f}")


# ---------------------------
# Step 11: Generate Parity Plots for Each Split
# ---------------------------

# Function to generate parity plot
def generate_parity_plot(train_df, val_df, test_df):
    plt.figure(figsize=(8, 8))
    
    # Plot Training data in Blue
    plt.scatter(x='target', y='predicted_Cp', data=train_df, color='blue', label='Training', alpha=0.5)
    
    # Plot Validation data in Red
    plt.scatter(x='target', y='predicted_Cp', data=val_df, color='red', label='Validation', alpha=0.5)
    
    # Plot Testing data in Green
    plt.scatter(x='target', y='predicted_Cp', data=test_df, color='green', label='Testing', alpha=0.5)
    
    # Calculate min and max values across all datasets for the reference line
    max_val = max(train_df['target'].max(), train_df['predicted_Cp'].max(),
                  val_df['target'].max(), val_df['predicted_Cp'].max(),
                  test_df['target'].max(), test_df['predicted_Cp'].max())
    min_val = min(train_df['target'].min(), train_df['predicted_Cp'].min(),
                  val_df['target'].min(), val_df['predicted_Cp'].min(),
                  test_df['target'].min(), test_df['predicted_Cp'].min())
    
    # Plot the y = x reference line
    plt.plot([min_val, max_val], [min_val, max_val], 'k--',)
    
    # Set labels and title
    plt.xlabel('Target Cp (kJ/kmol⋅K)', fontsize=12)
    plt.ylabel('Predicted Cp (kJ/kmol⋅K)', fontsize=12)
    plt.title('Parity Plot for C_p in decoupled multitask (AGC)', fontsize=14)
    
    # Add legend
    plt.legend()
    
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

# Call the unified parity plot function
generate_parity_plot(train_df, val_df, test_df)

# ---------------------------
# Optional Step 12: Save the Split DataFrames
# ---------------------------

# Save the split DataFrames to CSV files
train_df.to_csv("env/train_df.csv", index=False)
val_df.to_csv("env/val_df.csv", index=False)
test_df.to_csv("env/test_df.csv", index=False)

print("\nSplit DataFrames have been saved to the 'env' directory.")
