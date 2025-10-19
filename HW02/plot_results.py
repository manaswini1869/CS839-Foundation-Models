import pandas as pd
import matplotlib.pyplot as plt
import re

# Load the evaluation results
try:
    df = pd.read_csv('twain_evaluation_results.csv')
except FileNotFoundError:
    print("Error: evaluation_results.csv not found. Please run the evaluation script first.")
    exit()

# Function to extract character count from the model name (e.g., 'out-twain-50k')
def get_chars(model_name):
    match = re.search(r'-(\d+)k', model_name)
    if match:
        return int(match.group(1)) * 1000
    return 0

# Apply the function to create a new column for character count
df['train_chars'] = df['model'].apply(get_chars)

# Sort the dataframe by character count
df = df.sort_values('train_chars').reset_index(drop=True)

print("Evaluation Data:")
print(df)

# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Trigram Overlap on the primary y-axis
color = 'tab:blue'
ax1.set_xlabel('Number of Training Characters')
ax1.set_ylabel('Trigram Overlap (%)', color=color)
ax1.plot(df['train_chars'], df['trigram_overlap'], color=color, marker='o', label='Trigram Overlap')
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis for Word Validity
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Word Validity (%)', color=color)
ax2.plot(df['train_chars'], df['word_validity'], color=color, marker='s', linestyle='--', label='Word Validity')
ax2.tick_params(axis='y', labelcolor=color)

# Final plot adjustments
plt.title('Model Performance vs. Training Data Size (Mark Twain)')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_xscale('log') # Use a log scale for the x-axis for better visualization
fig.tight_layout()

# Save the plot
plt.savefig('training_scaling_plot.png')
print("\nPlot saved to training_scaling_plot.png")

# Show the plot
plt.show()
