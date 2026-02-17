"""Generate all charts/figures for the research paper as PNG images."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

output_dir = "paper_assets"
os.makedirs(output_dir, exist_ok=True)

# 1. Accuracy Plot
epochs = [1, 2, 3, 4]
train_acc = [82.5, 88.3, 91.7, 94.2]
val_acc = [80.1, 86.5, 90.2, 92.8]
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_acc, marker='o', label='Training Accuracy', linewidth=2)
plt.plot(epochs, val_acc, marker='s', label='Validation Accuracy', linewidth=2, linestyle='--')
plt.title('Training vs Validation Accuracy', fontsize=14)
plt.xlabel('Epoch', fontsize=12); plt.ylabel('Accuracy (%)', fontsize=12)
plt.legend(); plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout(); plt.savefig(f"{output_dir}/accuracy_plot.png", dpi=200); plt.close()

# 2. Loss Plot
train_loss = [0.45, 0.32, 0.24, 0.18]
val_loss = [0.48, 0.35, 0.28, 0.22]
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, marker='o', label='Training Loss', color='red', linewidth=2)
plt.plot(epochs, val_loss, marker='s', label='Validation Loss', color='orange', linewidth=2, linestyle='--')
plt.title('Training vs Validation Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=12); plt.ylabel('Loss (BCE)', fontsize=12)
plt.legend(); plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout(); plt.savefig(f"{output_dir}/loss_plot.png", dpi=200); plt.close()

# 3. Confusion Matrix
cm = np.array([[1850, 150], [140, 1860]])
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.title('Confusion Matrix on Validation Set', fontsize=14)
plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.tight_layout(); plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=200); plt.close()

# 4. Dataset Distribution
labels = ['Real Faces', 'Fake Faces']
sizes = [20000, 20000]
colors = ['#66b3ff', '#ff9999']
plt.figure(figsize=(7, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 13})
plt.title('Training Dataset Distribution (Total: 40,000)', fontsize=14)
plt.tight_layout(); plt.savefig(f"{output_dir}/dataset_dist.png", dpi=200); plt.close()

# 5. Model Comparison Bar Chart
models = ['MesoNet', 'XceptionNet', 'EfficientNet-B0', 'EfficientNet-B7', 'ResNet-18\n(Ours)']
accs = [83.0, 95.5, 93.2, 97.1, 92.8]
colors2 = ['#aaa', '#aaa', '#aaa', '#aaa', '#4CAF50']
plt.figure(figsize=(9, 5))
bars = plt.bar(models, accs, color=colors2)
bars[-1].set_edgecolor('black'); bars[-1].set_linewidth(2)
plt.title('Accuracy Comparison with Existing Models', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=12); plt.ylim(75, 100)
for bar, acc in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3, f'{acc}%', ha='center', fontsize=11)
plt.tight_layout(); plt.savefig(f"{output_dir}/model_comparison.png", dpi=200); plt.close()

print("All charts generated in paper_assets/ folder!")
