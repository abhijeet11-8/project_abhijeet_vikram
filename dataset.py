from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

class classfinder(ImageFolder):
    def find_classes(self, directory):
        # Define the fixed class order
        fixed_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        classes = [d.name for d in os.scandir(directory) if d.is_dir() and d.name in fixed_classes]
        classes.sort(key=lambda x: fixed_classes.index(x))  # Sort based on the fixed order
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(fixed_classes)}
        return classes, class_to_idx

def the_dataloader(root, batch_size=64, shuffle=True):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = classfinder(root=root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset.classes