import matplotlib.pyplot as plt
import pandas as pd

# Load CSV
df = pd.read_csv("report/training_losses.csv")

# Clean column names for easy access
col_text_only = "training-starcoder2-3b-python-text_only - train/sft_loss"
col_formal_sft = "training-starcoder2-3b-python-formal_sft - train/sft_loss"
col_woft = "training-starcoder2-3b-python - train/iwae_loss"


# Helper for running average smoothing (window=250)
def smooth(series, window=250):
    return series.rolling(window=window, min_periods=1).mean()


plt.figure(figsize=(10, 6), dpi=300)

# Plot smoothed bold lines only
plt.plot(df["Step"], smooth(df[col_formal_sft]), color="#ff7f0e", label="Text+Formal SFT", linewidth=2)
plt.plot(df["Step"], smooth(df[col_text_only]), color="#1f77b4", label="Text SFT", linewidth=2)
plt.plot(df["Step"], smooth(df[col_woft]), color="#2ca02c", label="WoFT Fine-Tuning", linewidth=2)

plt.title("Training Loss Comparison", fontsize=16, fontweight="bold", pad=15)
plt.xlabel("Training Steps", fontsize=14, labelpad=10)
plt.ylabel("Loss", fontsize=14, labelpad=10)
plt.ylim(0.5, 2)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12, loc="upper right")

plt.tight_layout()

# Save PDF
plt.savefig("report/training_losses.pdf", dpi=300)

print("Plots saved successfully.")
