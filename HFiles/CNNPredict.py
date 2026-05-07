import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans

from collections import Counter
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries, quickshift

from StandardSarCNN import StandardSarCNN


BASE_DIR = r"C:\Users\hmull\CSC2053\RESEARCHTHREE\RESEARCHTHREEOUTPUTS\ES"
MODEL_PATH = os.path.join(BASE_DIR, "ES_model_v1.pt")
DATA_DIR = os.path.join(BASE_DIR, "test")

BATCH_SIZE = 32

OUTPUT_FOLDER = os.path.join(BASE_DIR, "Output", "GradCAM")
LIME_OUTPUT_FOLDER = os.path.join(BASE_DIR, "Output", "LIME")
UMAP_OUTPUT_FOLDER = os.path.join(BASE_DIR, "Output", "UMAP")

LIME_IMAGE_COUNT = 10
LIME_NUM_SAMPLES = 300
LIME_NUM_FEATURES = 8

# =========================================


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_index=None):
        input_tensor = input_tensor.requires_grad_(True)

        output = self.model(input_tensor)

        if class_index is None:
            class_index = torch.argmax(output, dim=1).item()
        elif isinstance(class_index, torch.Tensor):
            class_index = int(class_index.item())

        self.model.zero_grad()

        loss = output[:, class_index]
        loss.backward()

        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * activations, dim=1)
        cam = F.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min != 0:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam




def overlay_gradcam(image, cam):
    cam = cv2.resize(cam, (128, 128))

    cam_uint8 = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    image = np.array(image.resize((128, 128)).convert("RGB"))
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = heatmap * 0.4 + image

    return overlay.astype(np.uint8)

def extract_features(model, loader, device):
    model.eval()

    features_list = []
    labels_list = []

    penultimate_features = None

    def hook_fn(module, input, output):
        nonlocal penultimate_features
        penultimate_features = output

    # hook fc1 (penultimate layer)
    hook = model.fc1.register_forward_hook(hook_fn)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            _ = model(images)  # forward pass triggers hook

            features_list.append(penultimate_features.cpu())
            labels_list.append(labels)

    hook.remove()

    X = torch.cat(features_list, dim=0).numpy()
    y = torch.cat(labels_list, dim=0).numpy()

    return X, y

def run_umap_analysis(model, loader, device, classes):
    print("\nExtracting features for UMAP...")
    X, y = extract_features(model, loader, device)

    print("Feature shape:", X.shape)

    print("Running UMAP...")
    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=20,
        min_dist=0.1,
        metric="euclidean",
        random_state=42
    )

    X_umap = umap_model.fit_transform(X)

    print("Running KMeans clustering...")
    k = len(classes)  # or manually set like k=5
    kmeans = KMeans(n_clusters=k, random_state=42)

    cluster_labels = kmeans.fit_predict(X_umap)

    for k in set(cluster_labels):
        if k == -1:
            continue

        idxs = np.where(cluster_labels == k)[0]
        counts = Counter(y[idxs])

        readable = {classes[i]: counts[i] for i in counts}
        print(f"Cluster {k}: size={len(idxs)}, labels={readable}")

    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=cluster_labels,
        cmap="Spectral",
        s=12
    )

    plt.title("UMAP + HDBSCAN Feature Clustering")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.colorbar(scatter, label="Cluster ID")

    save_path = os.path.join(UMAP_OUTPUT_FOLDER, "umap_clusters.png")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.tight_layout()
    
    plt.close()

    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    noise = np.sum(cluster_labels == -1)

    print("\n--- UMAP Results ---")
    print("Clusters found:", num_clusters)
    print("Noise points:", noise)
    print("Total samples:", len(cluster_labels))
    print("Saved:", save_path)


def save_lime_explanation(
    image_path,
    model,
    transform,
    predicted_class,
    save_path,
    device,
    num_classes,
    num_samples=LIME_NUM_SAMPLES,
    num_features=LIME_NUM_FEATURES
):
    """
    Generates and saves a LIME explanation for one image from disk.
    This is kept separate so Grad-CAM processing is not affected.
    """

    original_pil = Image.open(image_path).convert("RGB")
    base_image = np.array(original_pil.resize((128, 128)).convert("RGB"))

    explainer = lime_image.LimeImageExplainer()

    def classifier_fn(images):
        batch_tensors = []

        for img in images:
            img = np.clip(img, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img).convert("RGB")
            tensor = transform(pil_img)
            batch_tensors.append(tensor)

        batch = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    explanation = explainer.explain_instance(
        image=base_image,
        classifier_fn=classifier_fn,
        top_labels=num_classes,
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=lambda x: quickshift(x, kernel_size=4, max_dist=200, ratio=0.2)
    )

    temp, mask = explanation.get_image_and_mask(
        label=int(predicted_class),
        positive_only=True,
        num_features=num_features,
        hide_rest=False
    )

    temp = temp.astype(np.float32)
    if temp.max() > 1.0:
        temp = temp / 255.0

    lime_vis = mark_boundaries(temp, mask)
    lime_vis = (lime_vis * 255).astype(np.uint8)

    cv2.imwrite(save_path, cv2.cvtColor(lime_vis, cv2.COLOR_RGB2BGR))


def main():
    try:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(LIME_OUTPUT_FOLDER, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        transform = StandardSarCNN.transform()

        dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        classes = dataset.classes
        num_classes = len(classes)

        model = StandardSarCNN(num_classes)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()

        gradcam = GradCAM(model, model.conv8)

        result_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

        testing_correct = 0
        testing_total = 0

        image_counter = 0

        # Choose 5 random dataset indices for LIME
        total_images = len(dataset.samples)
        lime_count = min(LIME_IMAGE_COUNT, total_images)
        lime_indices = set(random.sample(range(total_images), lime_count))

        lime_jobs = []

        for batch_index, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

            testing_total += labels.size(0)
            testing_correct += (predicted == labels).sum().item()

            for i in range(len(labels)):
                actual_label = int(labels[i].item())
                predicted_label = int(predicted[i].item())
                result_matrix[actual_label][predicted_label] += 1

            # GradCAM for all images
            for i in range(len(images)):
                input_tensor = images[i].unsqueeze(0)

                with torch.enable_grad():
                    cam = gradcam.generate(input_tensor, predicted[i])

                original = transforms.ToPILImage()(images[i].detach().cpu())

                overlay = overlay_gradcam(original, cam)

                filename = (
                    f"{image_counter}_actual-{classes[labels[i].item()]}"
                    f"_pred-{classes[predicted[i].item()]}.png"
                )

                save_path = os.path.join(OUTPUT_FOLDER, filename)

                cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                if image_counter in lime_indices:
                    image_path, _ = dataset.samples[image_counter]
                    lime_filename = (
                        f"{image_counter}_actual-{classes[labels[i].item()]}"
                        f"_pred-{classes[predicted[i].item()]}_lime.png"
                    )
                    lime_save_path = os.path.join(LIME_OUTPUT_FOLDER, lime_filename)

                    lime_jobs.append(
                        {
                            "image_path": image_path,
                            "predicted_class": predicted[i].item(),
                            "save_path": lime_save_path
                        }
                    )

                image_counter += 1

        # Currently Running LIME only on the 5 selected images
        print(f"\nGenerating LIME for {len(lime_jobs)} random images...")

        for job in lime_jobs:
            save_lime_explanation(
                image_path=job["image_path"],
                model=model,
                transform=transform,
                predicted_class=job["predicted_class"],
                save_path=job["save_path"],
                device=device,
                num_classes=num_classes
            )
        run_umap_analysis(model, data_loader, device, classes)

        testing_accuracy = testing_correct / testing_total * 100
        print(f"\nOverall accuracy on dataset: {testing_accuracy:.2f}%\n")

        padding = len(max(classes, key=len)) + 4
        header = " " * padding + "".join(f"{name:>{padding}}" for name in classes)
        print(header)

        for i in range(num_classes):
            row = f"{classes[i]:{padding}}"
            for j in range(num_classes):
                row += f"{result_matrix[i][j]:{padding}}"
            print(row)

        print("\nPer-class metrics:\n")

        for i in range(num_classes):
            TP = result_matrix[i][i]
            FN = sum(result_matrix[i]) - TP
            FP = sum(result_matrix[j][i] for j in range(num_classes)) - TP

            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0

            print(
                f"{classes[i]:15} | "
                f"Precision: {precision:.4f} | "
                f"Recall: {recall:.4f} | "
                f"F1 Score: {f1:.4f}"
            )

        print(f"\nGradCAM images saved to:\n{OUTPUT_FOLDER}")
        print(f"LIME images saved to:\n{LIME_OUTPUT_FOLDER}")


    except Exception as e:
        print("Error:", e)
        exit(1)


if __name__ == "__main__":
    main()