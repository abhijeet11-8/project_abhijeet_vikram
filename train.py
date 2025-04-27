import torch
import os
import random
import shutil
from dataset import the_dataloader  # Assuming dataset.py contains the get_dataloader function

def select_random_images(source_dir, temp_dir, num_images=10):
    """
    Select 10 random images for each class and copy them to a temporary directory.
    Args:
        source_dir (str): Path to the source directory (e.g., /Users/admin/Desktop/FCR2013/data/train).
        temp_dir (str): Path to the temporary directory for selected images.
        num_images (int): Number of images to select per class.
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for emotion_class in os.listdir(source_dir):
        class_source = os.path.join(source_dir, emotion_class)
        class_dest = os.path.join(temp_dir, emotion_class)

        # Skip non-directory files (e.g., .DS_Store)
        if not os.path.isdir(class_source):
            continue

        if not os.path.exists(class_dest):
            os.makedirs(class_dest)

        # Get all image files in the emotion class folder
        image_files = [f for f in os.listdir(class_source) if os.path.isfile(os.path.join(class_source, f))]

        # Randomly select `num_images` images
        selected_images = random.sample(image_files, min(num_images, len(image_files)))

        # Copy selected images to the temporary directory
        for image in selected_images:
            src_path = os.path.join(class_source, image)
            dest_path = os.path.join(class_dest, image)
            shutil.copy(src_path, dest_path)

def the_trainer(model, train_loader, loss_fn, optimizer, device, num_classes, save_path="checkpoints/final_weights.pth"):
    model.to(device)
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.detach().item() * inputs.size(0)

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        for i in range(len(labels)):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                class_correct[label] += 1
            class_total[label] += 1

    avg_loss = running_loss / len(train_loader.dataset)
    accuracy = 100.0 * correct / total

    print(f"Train Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # Class-wise accuracy
    for i in range(num_classes):
        if class_total[i] != 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            print(f"  Class {i} Accuracy: {acc:.2f}%")
        else:
            print(f"  Class {i} Accuracy: N/A")

    # Save the model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Temporary directory for selected images
    temp_dir = "/Users/admin/Desktop/FCR2013/temp_train"

    # Select 10 random images per class
    source_directory = "/Users/admin/Desktop/FCR2013/data/train"
    select_random_images(source_directory, temp_dir, num_images=10)

    # Load the model
    from model import TheModel  # Assuming model_2.py contains the model
    model = TheModel().to(device)

    # Load training data from the temporary directory
    train_loader, _ = the_dataloader(temp_dir, batch_size=128, shuffle=True)

    # Define loss function and optimizer
    from config import FocalLoss  # Assuming FocalLoss is defined in config.py
    loss_fn = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    the_trainer(
        model=model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_classes=7,
        save_path="/Users/admin/Desktop/FCR2013/best_model_weights.pth"
    )

    # Clean up the temporary directory after training
    shutil.rmtree(temp_dir)
    print(f"Temporary directory {temp_dir} removed.")