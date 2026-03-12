import argparse
import os
from typing import List, Dict

from pathlib import Path

import pickle
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

from Models.GetModel import get_model


def main():
    try:
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        parser.add_argument("model_name",
                            type=str,
                            help="Model name.")
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
        parser.add_argument("--min_epochs",
                            type=int,
                            help="Number of epochs forced despite patience suggesting otherwise.",
                            required=False,
                            default=-1)
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
        parser.add_argument("--weight_decay",
                            type=float,
                            help="Number of no improvement epochs that will be tolerated.",
                            required=False,
                            default=0.0)
        parser.add_argument("--scheduler_step_size",
                            type=float,
                            help="The number epochs after which learning rate will be reduced.",
                            required=False,
                            default=10.0)
        parser.add_argument("--learning_rate_decay",
                            type=float,
                            help="The learning rate decay corresponding with scheduler_step_size.",
                            required=False,
                            default=1.0)
        parser.add_argument("--max_gradient_norm",
                            type=float,
                            help="Maximal norm for gradient clipping.",
                            required=False,
                            default=10.0)
        parser.add_argument("--out_folder",
                            type=str,
                            help="Output folder path.",
                            required=False,
                            default="Output")
        parser.add_argument("--testing_mode",
                            type=bool,
                            help="Enables some weird parameters...",
                            required=False,
                            default=False)
        args = parser.parse_args()
        model_name: str = args.model_name
        out_root: str = args.out_name
        train_directory: str = args.train_dir
        test_directory: str = args.test_dir
        num_epochs: int = args.num_epochs
        min_epochs: int = args.min_epochs
        batch_size: int = args.batch_size
        learning_rate: float = args.learning_rate
        patience: int = args.patience
        weight_decay: float = args.weight_decay
        scheduler_step_size: int = args.scheduler_step_size
        learning_rate_decay: float = args.learning_rate_decay
        max_gradient_norm: float = args.max_gradient_norm
        out_directory: str = args.out_folder
        testing_mode: bool = args.testing_mode

        train_dir = Path(train_directory)
        num_classes = sum(1 for item in train_dir.iterdir() if item.is_dir())
        model = get_model(model_name, num_classes)

        def init_matlab_he(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if testing_mode:
            model.apply(init_matlab_he)

        train_dataset = datasets.ImageFolder(root=train_directory, transform=model.train_transform)
        test_dataset = datasets.ImageFolder(root=test_directory, transform=model.test_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Loss & optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=learning_rate_decay)

        start = time.time()

        # Training graph data
        history_training_accuracy: List[float] = []
        history_training_loss: List[float] = []
        history_validation_accuracy: List[float] = []
        history_validation_loss: List[float] = []

        history_training_accuracy_hf: List[float] = []
        history_training_loss_hf: List[float] = []
        history_validation_accuracy_hf: List[float] = []
        history_validation_loss_hf: List[float] = []

        saved_roots : List[str] = []
        saved_epoch : Dict[str, int] = {}
        def save_model(save_epoch: int, overwrite: bool = False):
            if overwrite:
                for saved_root in saved_roots:
                    if os.path.exists(f"{saved_root}.pt"):
                        os.remove(f"{saved_root}.pt")
                saved_roots.clear()
                saved_epoch.clear()

            os.makedirs(f"{out_directory}", exist_ok=True)
            save_root = f"{out_directory}/{out_root}_{len(saved_roots) + 1}"
            torch.save(model.state_dict(), f"{save_root}.pt")
            saved_roots.append(save_root)
            saved_epoch[f"{out_root}_{len(saved_roots) + 1}"] = save_epoch

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

                _, predicted = torch.max(outputs, 1)
                iteration_correct = (predicted == labels).sum().item()
                iteration_total = labels.size(0)
                iteration_loss = loss.item()

                training_loss += iteration_loss
                training_correct += iteration_correct
                training_total += iteration_total

                # Save training iteration data
                history_training_accuracy_hf.append(iteration_correct / iteration_total * 100)
                history_training_loss_hf.append(iteration_loss)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_gradient_norm)
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

                    _, predicted = torch.max(outputs, 1)
                    iteration_correct = (predicted == labels).sum().item()
                    iteration_total = labels.size(0)
                    iteration_loss = loss.item()

                    validation_correct += iteration_correct
                    validation_total += iteration_total
                    validation_loss += iteration_loss

                    # Save testing iteration data
                    history_validation_accuracy_hf.append(iteration_correct / iteration_total * 100)
                    history_validation_loss_hf.append(iteration_loss)
            epoch_validation_accuracy = validation_correct / validation_total * 100
            epoch_validation_loss = validation_loss / len(test_loader)
            history_validation_accuracy.append(epoch_validation_accuracy)
            history_validation_loss.append(epoch_validation_loss)

            print(f"[Epoch {epoch + 1} for {out_root}] "
                  f"Accuracy: {epoch_training_accuracy:.2f}%, Loss: {epoch_training_loss:.4f}, "
                  f"V-Accuracy: {epoch_validation_accuracy:.2f}%, V-Loss: {epoch_validation_loss:.4f}")

            if epoch_validation_loss <= best_validation_loss:
                best_validation_loss = epoch_validation_loss
                epoch_since_last_improve = 0
                save_model(epoch, True)
            else:
                save_model(epoch)
                epoch_since_last_improve += 1
                if (min_epochs <= 0 or epoch >= min_epochs) and epoch_since_last_improve == patience:
                    print(f"Patience threshold met, terminating training.")
                    break

            scheduler.step()

        end = time.time()
        elapsed = end - start
        elapsed_hours = int(elapsed // 3600)
        elapsed_minutes = int(elapsed % 3600 // 60)
        elapsed_seconds = round(elapsed % 60, 2)
        print(f"Training completed in {elapsed_hours}:{elapsed_minutes}:{elapsed_seconds}, saved: {len(saved_roots) + 1} models.")

        history = {
            "epoch_training_accuracy": history_training_accuracy,
            "epoch_training_loss": history_training_loss,
            "epoch_validation_accuracy": history_validation_accuracy,
            "epoch_validation_loss": history_validation_loss,
            "iteration_training_accuracy": history_training_accuracy_hf,
            "iteration_training_loss": history_training_loss_hf,
            "iteration_validation_accuracy": history_validation_accuracy_hf,
            "iteration_validation_loss": history_validation_loss_hf,
            "saved_epochs": saved_epoch,
        }

        with open(f"{out_directory}/{out_root}.pkl", "wb") as file:
            pickle.dump(history, file)

    except Exception as exception:
        print(exception)
        exit(1)

if __name__ == "__main__":
    main()