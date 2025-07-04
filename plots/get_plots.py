from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = Path(__file__).resolve().parent # root/plots

# Load the CSV data
df = pd.read_csv(f"{PLOTS_DIR}/../output/validation_results.csv", sep=";")

# Drop columns not needed for plotting
df = df.drop(columns=["TP", "FP", "FN"])

### SCRIPT 1: F1-score by technique and temperature (grouped bar plot) ###

# Calculate mean metrics grouped by technique and temperature
grouped = df.groupby(["technique", "temperature"]).mean(numeric_only=True).reset_index()

techniques = grouped["technique"].unique()
temperatures = sorted(grouped["temperature"].unique())

bar_width = 0.2
x = np.arange(len(techniques))

fig1, ax1 = plt.subplots(figsize=(12, 6))

colors = ['tab:blue', 'tab:orange', 'tab:green']

# Plot grouped bars for each temperature
for i, temp in enumerate(temperatures):
    temp_data = grouped[grouped["temperature"] == temp]
    temp_data = temp_data.set_index('technique').reindex(techniques).reset_index()
    positions = x + i * bar_width
    ax1.bar(positions, temp_data["f1-score"], width=bar_width, color=colors[i], label=f'Temperatura {temp}')

# Configure plot labels and styles
ax1.set_xticks(x + bar_width)
ax1.set_xticklabels(techniques, rotation=45)
ax1.tick_params(axis='both', labelsize=18)
ax1.set_xlabel('Technique', fontsize=20)
ax1.set_ylabel('F1-score', fontsize=20)
ax1.set_title('F1-score by technique and temperature', fontsize=26)
ax1.legend()

plt.tight_layout()
fig1.savefig(f"{PLOTS_DIR}/f1score_technique_temperature.png", dpi=300, bbox_inches="tight")
plt.close(fig1)

### SCRIPT 2: Heatmap of recall by technique and temperature (no seaborn) ###

# Pivot table with recall values
pivot = df.pivot_table(index="technique", columns="temperature", values="recall")

fig2, ax2 = plt.subplots(figsize=(8,5))
data = pivot.values
techniques = pivot.index.tolist()
temperatures = pivot.columns.tolist()

# Draw the heatmap
cax = ax2.imshow(data, cmap='Blues', aspect='auto')

# Configure axes
ax2.set_xticks(np.arange(len(temperatures)))
ax2.set_yticks(np.arange(len(techniques)))
ax2.set_xticklabels(temperatures)
ax2.set_yticklabels(techniques)

plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add numeric labels inside heatmap cells
for i in range(len(techniques)):
    for j in range(len(temperatures)):
        ax2.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", color="black")

# Add titles and colorbar
ax2.set_title("Recall heatmap by technique and temperature", fontsize=16)
ax2.set_xlabel("Temperature")
ax2.set_ylabel("Technique")
fig2.colorbar(cax, ax=ax2)

plt.tight_layout()
fig2.savefig(f"{PLOTS_DIR}/recall_heatmap.png", dpi=300, bbox_inches="tight")
plt.close(fig2)

### SCRIPT 3: Grouped bars for precision, recall, and F1-score by technique and temperature ###

# Compute average metrics by temperature and by technique
grouped_by_temperature = df.groupby("temperature").mean(numeric_only=True)[["precision", "recall", "f1-score"]]
grouped_by_technique = df.groupby("technique").mean(numeric_only=True)[["precision", "recall", "f1-score"]]

def plot_grouped_bars(data, group_label, title, filename):
    metrics = data.columns.tolist()
    groups = data.index.tolist()
    x = np.arange(len(groups))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10,6))
    # Draw grouped bars for each metric
    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, data[metric], width, label=metric)

    # Configure plot
    ax.set_xticks(x + width)
    ax.set_xticklabels(groups, rotation=45)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylabel('Average value', fontsize=16)
    ax.set_xlabel(group_label, fontsize=16)
    ax.set_title(title, fontsize=20)
    ax.legend(fontsize=14)
    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

# Generate plots for technique and temperature
plot_grouped_bars(grouped_by_technique, "Technique", "Average Precision, Recall and F1-score by Technique", f"{PLOTS_DIR}/metrics_by_technique.png")
plot_grouped_bars(grouped_by_temperature, "Temperature", "Average Precision, Recall and F1-score by Temperature", f"{PLOTS_DIR}/metrics_by_temperature.png")
