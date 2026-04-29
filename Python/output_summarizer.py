import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

history_file = r"Output\raw_outputs\python_cnn_output_decibel_30-70.pkl"
save_dir = r"Output\plots\30-70\decibel"
os.makedirs(save_dir, exist_ok=True)

with open(history_file, "rb") as f:
    history = pickle.load(f)

train_acc = np.array(history["epoch_training_accuracy"])
val_acc = np.array(history["epoch_validation_accuracy"])
train_loss = np.array(history["epoch_training_loss"])
val_loss = np.array(history["epoch_validation_loss"])

best_epoch = np.argmin(val_loss)
best_val_acc = val_acc[best_epoch]

plt.figure(figsize=(8,5))
plt.plot(train_acc, label="Train Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.axvline(best_epoch, color='r', linestyle='--', label="Best Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training & Validation Accuracy")
plt.legend()
acc_plot_path = os.path.join(save_dir, "decibel_accuracy_plot.png")
plt.savefig(acc_plot_path)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="Validation Loss")
plt.axvline(best_epoch, color='r', linestyle='--', label="Best Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
loss_plot_path = os.path.join(save_dir, "decibel_loss_plot.png")
plt.savefig(loss_plot_path)
plt.show()

print(f"Plots saved to {save_dir}")
print(f"Total epochs run: {len(train_acc)}")
print(f"Final training accuracy: {train_acc[-1]:.2f}%")
print(f"Final validation accuracy: {val_acc[-1]:.2f}%")
print(f"Best epoch: {best_epoch+1}")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print(f"Training loss at best epoch: {train_loss[best_epoch]:.4f}")
print(f"Validation loss at best epoch: {val_loss[best_epoch]:.4f}")