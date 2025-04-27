import os
import zipfile
import torch
from torchvision import transforms
from PIL import Image
from model import TheModel  # Assuming the model is defined in model.py

# Define the emotion classes
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Define the image preprocessing pipeline
def preprocess_image(image_path):
    """
    Preprocess a single image for the model.
    Args:
        image_path (str): Path to the image file.
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((48, 48)),                 # Resize to 48x48
        transforms.ToTensor(),                       # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))         # Normalize to [-1, 1]
    ])
    image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB mode
    return transform(image)

# Function to unzip the data if it is in a zip file
def unzip_data(zip_path, extract_to):
    """
    Unzip the data file if it exists.
    Args:
        zip_path (str): Path to the zip file.
        extract_to (str): Directory to extract the contents to.
    """
    print(f"Unzipping data from {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Data extracted to {extract_to}")

# Function to load all images from a directory with subfolders
def load_images_from_directory(data_dir):
    """
    Load all images from a directory with subfolders for each class.
    Args:
        data_dir (str): Path to the directory containing subfolders for each class.
    Returns:
        list: List of tuples (image_path, label).
    """
    image_paths = []
    for emotion_class in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, emotion_class)
        if os.path.isdir(class_dir):
            for image_file in os.listdir(class_dir):
                # Only process .jpg files
                if image_file.lower().endswith(".jpg"):
                    image_path = os.path.join(class_dir, image_file)
                    image_paths.append((image_path, emotion_class))  # Include the label (folder name)
    return image_paths

# Define the classification function
def classify_images(list_of_img_paths, model_weights_path="final_weights_new.pth"):
    """
    Classify a batch of images using the trained model.
    Args:
        list_of_img_paths (list): List of image file paths.
        model_weights_path (str): Path to the trained model weights.
    Returns:
        dict: Dictionary mapping image paths to predicted emotion labels.
    """
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TheModel(num_classes=len(EMOTION_CLASSES)).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    # Preprocess the images
    images = torch.stack([preprocess_image(img_path) for img_path, _ in list_of_img_paths]).to(device)

    # Perform inference
    with torch.no_grad():
        logits = model(images)
        predictions = torch.argmax(logits, dim=1)

    # Map predictions to emotion labels
    results = {img_path: EMOTION_CLASSES[pred] for (img_path, _), pred in zip(list_of_img_paths, predictions)}
    return results, predictions

if __name__ == "__main__":
    # Path to the input file (zip or directory)
    input_path = os.path.join(os.getcwd(), "test")  # Relative path to the `test` directory
    extraction_directory = os.path.join(os.getcwd(), "extracted_data")  # Relative path to the `extracted_data` directory
    model_weights_path = os.path.join(os.getcwd(), "final_weights_new.pth")  # Relative path to the weights file

    # Check if the input is a zip file or a directory
    if zipfile.is_zipfile(input_path):
        # If it's a zip file, unzip it
        unzip_data(input_path, extraction_directory)
        test_directory = os.path.join(extraction_directory, "test")  # Test directory inside the extracted data
    elif os.path.isdir(input_path):
        # If it's already a directory, use it directly
        test_directory = input_path
    else:
        raise ValueError(f"Invalid input path: {input_path}. Must be a zip file or a directory.")

    # Check if the test directory exists
    if not os.path.exists(test_directory):
        raise FileNotFoundError(f"Test directory not found: {test_directory}")

    # Load all images from the test directory
    test_image_paths = load_images_from_directory(test_directory)

    # Classify the images
    predictions, predicted_indices = classify_images(test_image_paths, model_weights_path)

    # Calculate accuracy
    correct = 0
    total = len(test_image_paths)
    for (img_path, true_label), predicted_index in zip(test_image_paths, predicted_indices):
        predicted_label = EMOTION_CLASSES[predicted_index]
        if predicted_label == true_label:
            correct += 1
        print(f"Image: {img_path} -> Predicted: {predicted_label}, True: {true_label}")

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\nAccuracy: {accuracy:.2f}%")