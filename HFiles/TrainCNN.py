import os
from typing import List
import pickle
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

from StandardSarCNN import StandardSarCNN


# =========================================
# 🔧 CHANGE SETTINGS HERE ONLY
# =========================================

OUT_NAME = "EC_model_v1"

BASE_DIR = r"C:\Users\hmull\OneDrive\Documents\CSC2053\RESEARCHTHREE"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

NUM_EPOCHS = 10 #!!!!!!!!!!!!!!!!!!!!
BATCH_SIZE = 48
LEARNING_RATE = 0.001
PATIENCE = 5
OUT_FOLDER = os.path.join(BASE_DIR, "Output")
TEST_PARAMETERS = False   # True enables scheduler, grad clipping, weight decay

# =========================================


def main():
    try:
        transform = StandardSarCNN.transform()

        train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
        test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        num_classes = len(train_dataset.classes)
        model = StandardSarCNN(num_classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=(1e-4 if TEST_PARAMETERS else 0.0)
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )

        start = time.time()

        history_training_accuracy: List[float] = []
        history_training_loss: List[float] = []
        history_validation_accuracy: List[float] = []
        history_validation_loss: List[float] = []

        best_validation_loss = float("inf")
        epoch_since_last_improve = 0

        for epoch in range(NUM_EPOCHS):
            model.train()
            training_loss = 0.0
            training_correct = 0
            training_total = 0

            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()

                if TEST_PARAMETERS:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                optimizer.step()

                training_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                training_correct += (predicted == labels).sum().item()
                training_total += labels.size(0)

            epoch_training_accuracy = training_correct / training_total * 100
            epoch_training_loss = training_loss / len(train_loader)

            history_training_accuracy.append(epoch_training_accuracy)
            history_training_loss.append(epoch_training_loss)

            # Validation
            model.eval()
            validation_loss = 0.0
            validation_correct = 0
            validation_total = 0

            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    validation_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    validation_correct += (predicted == labels).sum().item()
                    validation_total += labels.size(0)

            epoch_validation_accuracy = validation_correct / validation_total * 100
            epoch_validation_loss = validation_loss / len(test_loader)

            history_validation_accuracy.append(epoch_validation_accuracy)
            history_validation_loss.append(epoch_validation_loss)

            print(
                f"[Epoch {epoch + 1}] "
                f"Train Acc: {epoch_training_accuracy:.2f}% | "
                f"Train Loss: {epoch_training_loss:.4f} | "
                f"Val Acc: {epoch_validation_accuracy:.2f}% | "
                f"Val Loss: {epoch_validation_loss:.4f}"
            )

            # Early stopping
            if epoch_validation_loss < best_validation_loss:
                best_validation_loss = epoch_validation_loss
                os.makedirs(OUT_FOLDER, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(OUT_FOLDER, f"{OUT_NAME}.pt"))
                epoch_since_last_improve = 0
            else:
                epoch_since_last_improve += 1
                if epoch_since_last_improve >= PATIENCE:
                    print("Patience threshold met. Stopping training.")
                    break

            if TEST_PARAMETERS:
                scheduler.step()

        end = time.time()
        elapsed = end - start
        print(f"\nTraining completed in {elapsed/60:.2f} minutes.")

        # Final test accuracy
        model.eval()
        testing_correct = 0
        testing_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                testing_total += labels.size(0)
                testing_correct += (predicted == labels).sum().item()

        testing_accuracy = testing_correct / testing_total * 100
        print(f"Final test accuracy: {testing_accuracy:.2f}%")

        # Save training history
        with open(os.path.join(OUT_FOLDER, f"{OUT_NAME}.pkl"), "wb") as file:
            pickle.dump(history_training_accuracy, file)
            pickle.dump(history_training_loss, file)
            pickle.dump(history_validation_accuracy, file)
            pickle.dump(history_validation_loss, file)

    except Exception as e:
        print("Error:", e)
        exit(1)


if __name__ == "__main__":
    main()