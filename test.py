import torch
from collections import Counter
from config import FocalLoss

# Function to get class weights for imbalance handling
def get_class_weights(test_loader, num_classes):
    label_counts = Counter()
    for _, labels in test_loader:
        label_counts.update(labels.tolist())

    total_samples = sum(label_counts.values())
    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 0)
        weights.append(total_samples / (num_classes * count) if count > 0 else 0)

    return torch.tensor(weights, dtype=torch.float)

# Test function to evaluate model performance
def test(model, test_loader, device, num_classes=7):
    model.to(device)
    model.eval()

    # Get class weights for loss calculation
    class_weights = get_class_weights(test_loader, num_classes).to(device)
    loss_fn = FocalLoss()  # Can be replaced by Focal Loss if needed

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = 100 * correct / total
    print(f"Test Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy