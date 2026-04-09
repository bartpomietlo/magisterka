import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Confusion matrix values
cm = np.array([[55, 40],  # TP, FN
               [4, 38]])  # FP, TN

# Labels
labels = ['AI', 'nie-AI']
pred_labels = ['Prognoza: AI', 'Prognoza: nie-AI']

# Create heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=pred_labels, yticklabels=labels, square=True,
            linewidths=0.5, linecolor='black')

plt.title('Globalna macierz pomyłek detektora (N=137)')
plt.ylabel('Faktycznie')
plt.xlabel('Prognoza')
plt.tight_layout()
plt.savefig('figures/confusion_matrix_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()