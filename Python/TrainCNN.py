import argparse
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

def main():
    try:
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        parser.add_argument("out_name",
                            type = str,
                            help = "The root name of output files.")
        parser.add_argument("train_dir",
                            type=str,
                            help="Path to a directory containing the training dataset.")
        parser.add_argument("test_dir",
                            type=str,
                            help="Path to a directory containing the testing dataset.")
        parser.add_argument("num_epochs",
                            type=int,
                            help="Number of training epochs.")
        parser.add_argument("--batch_size",
                            type=int,
                            help="Samples per batch.",
                            required=False,
                            default=48)
        parser.add_argument("--learning_rate",
                            type=float,
                            help="Initial learning rate.",
                            required=False,
                            default=0.001)
        parser.add_argument("--patience",
                            type=int,
                            help="Number of no improvement epochs that will be tolerated.",
                            required=False,
                            default=5)
        parser.add_argument("--out_folder",
                            type=str,
                            help="Output folder path.",
                            required=False,
                            default="Output")
        parser.add_argument("--test_parameters",
                            type=bool,
                            help="True enable learning weight scheduling, gradient clipping, and weight decay.",
                            required=False,
                            default=False)
        args = parser.parse_args()
        out_root: str = args.out_name
        train_directory: str = args.train_dir
        test_directory: str = args.test_dir
        num_epochs: int = args.num_epochs
        batch_size: int = args.batch_size
        learning_rate: float = args.learning_rate
        patience: int = args.patience
        out_directory: str = args.out_folder
        test_mode: bool = args.test_parameters

        transform = StandardSarCNN.transform()
        train_dataset = datasets.ImageFolder(root=train_directory, transform=transform)
        test_dataset = datasets.ImageFolder(root=test_directory, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        num_classes = len(train_dataset.classes)
        model = StandardSarCNN(num_classes)

        # Loss & optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=(1e-4 if test_mode else 0.0))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        start = time.time()

        # Training graph data
        history_training_accuracy: List[float] = []
        history_training_loss: List[float] = []
        history_validation_accuracy: List[float] = []
        history_validation_loss: List[float] = []

        best_validation_loss = float("inf")
        epoch_since_last_improve = 0

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            training_loss = 0.0
            training_correct = 0
            training_total = 0
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()

                training_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                training_correct += (predicted == labels).sum().item()
                training_total += labels.size(0)

                if test_mode:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            epoch_training_accuracy = training_correct / training_total * 100
            epoch_training_loss = training_loss / len(train_loader)
            history_training_accuracy.append(epoch_training_accuracy)
            history_training_loss.append(epoch_training_loss)

            # Evaluation
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

            print(f"[Epoch {epoch + 1} for {out_root}] "
                  f"Accuracy: {epoch_training_accuracy:.2f}%, Loss: {epoch_training_loss:.4f}, "
                  f"V-Accuracy: {epoch_validation_accuracy:.2f}%, V-Loss: {epoch_validation_loss:.4f}")

            if epoch_validation_loss < best_validation_loss:
                best_validation_loss = epoch_validation_loss
                os.makedirs(f"{out_directory}", exist_ok=True)
                torch.save(model.state_dict(), f"{out_directory}/{out_root}.pt")
                epoch_since_last_improve = 0
            else:
                if epoch_since_last_improve == patience:
                    print(f"Patience threshold met, terminating training.")
                    break
                epoch_since_last_improve += 1

            if test_mode:
                scheduler.step()

        end = time.time()
        elapsed = end - start
        print(f"Training completed in {elapsed / 60} minutes and {elapsed % 60} seconds.")

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
        print(f"Overall accuracy on test dataset: {testing_accuracy:.2f}%")

        with open(f"{out_directory}/{out_root}.pkl", "wb") as file:
            pickle.dump(history_training_accuracy, file)
            pickle.dump(history_training_loss, file)
            pickle.dump(history_validation_accuracy, file)
            pickle.dump(history_validation_loss, file)

    except Exception as exception:
        print(exception)
        exit(1)

if __name__ == "__main__":
    main()