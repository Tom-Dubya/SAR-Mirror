import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from StandardSarCNN import StandardSarCNN


def main():
    try:
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        parser.add_argument("model_dir",
                            type=str,
                            help="Path to the model.")
        parser.add_argument("data_dir",
                            type=str,
                            help="Path to a directory containing the dataset.")
        parser.add_argument("--batch_size",
                            type=int,
                            help="Samples per batch.",
                            required=False,
                            default=32)
        args = parser.parse_args()
        model_dir: str = args.model_dir
        data_dir: str = args.data_dir
        batch_size: int = args.batch_size

        transform = StandardSarCNN.transform()
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        classes = dataset.classes
        num_classes = len(classes)

        model = StandardSarCNN(num_classes)
        model.load_state_dict(torch.load(model_dir))
        model.eval()

        # Confusion-matrix-lite
        result_matrix: list = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
        testing_correct = 0
        testing_total = 0
        with torch.no_grad():
            for images, labels in data_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                testing_total += labels.size(0)
                testing_correct += (predicted == labels).sum().item()

                for i in range(len(labels)):
                    actual_label = int(labels[i])
                    predicted_label = int(predicted[i])
                    result_matrix[actual_label][predicted_label] += 1
        testing_accuracy = testing_correct / testing_total * 100
        print(f"Overall accuracy on dataset: {testing_accuracy:.2f}%")

        padding = len(max(classes, key=len)) + 2
        result_matrix_str: str = f"{"":{padding}}" + "".join([f"{name:>{padding}}" for name in classes])

        for i in range(num_classes):
            result_matrix_str += "\n"
            class_name = classes[i]
            result_matrix_str += f"{class_name:{padding}}"
            for j in range(num_classes):
                class_result = result_matrix[i][j]
                result_matrix_str += f"{class_result:{padding}}"
        print(result_matrix_str)

    except Exception as exception:
        print(exception)
        exit(1)

if __name__ == "__main__":
    main()