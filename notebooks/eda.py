import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="Set2")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eda_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)
COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

print("[EDA] Loading dataset...")
df = pd.read_csv(DATA_URL, names=COLUMNS)

print("\n--- Dataset Shape ---")
print(df.shape)
print("\n--- First 5 rows ---")
print(df.head())
print("\n--- Descriptive Statistics ---")
print(df.describe().round(2))
print("\n--- Class Distribution ---")
print(df["Outcome"].value_counts(normalize=True).map(lambda x: f"{x:.1%}"))

fig, ax = plt.subplots(figsize=(5, 4))
df["Outcome"].value_counts().plot(kind="bar", ax=ax, color=["#2ECC71","#E74C3C"], edgecolor="white", width=0.5)
ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
ax.set_xticklabels(["No Diabetes", "Diabetes"], rotation=0)
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_class_distribution.png"), dpi=150)
plt.close()

features = [c for c in COLUMNS if c != "Outcome"]
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for i, feat in enumerate(features):
    ax = axes[i // 3][i % 3]
    for outcome, color, label in [(0, "#2ECC71", "No Diabetes"), (1, "#E74C3C", "Diabetes")]:
        df[df["Outcome"] == outcome][feat].plot(kind="hist", bins=25, alpha=0.6,
                                                 ax=ax, color=color, label=label)
    ax.set_title(feat, fontweight="bold")
    ax.legend(fontsize=7)
plt.suptitle("Feature Distributions by Outcome", y=1.01, fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_feature_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(9, 7))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_correlation_heatmap.png"), dpi=150)
plt.close()

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for i, feat in enumerate(features):
    ax = axes[i // 4][i % 4]
    df.boxplot(column=feat, by="Outcome", ax=ax, patch_artist=True,
               boxprops=dict(facecolor="#AED6F1"),
               medianprops=dict(color="#E74C3C", linewidth=2))
    ax.set_title(feat, fontweight="bold")
    ax.set_xlabel("")
    ax.set_xticklabels(["No Diabetes", "Diabetes"])
plt.suptitle("Feature Boxplots by Outcome", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_boxplots.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\n[EDA] All plots saved to '{OUTPUT_DIR}/'")
print("[EDA] Done.")
