from pathlib import Path

from PIL import Image

from torch.utils.data import Dataset

from torchvision import transforms


class DeepfakeDataset(Dataset):

    def __init__(self, root):

        self.samples = []

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        real = Path(root) / "real"
        fake = Path(root) / "fake"

        for img in real.glob("*.jpg"):
            self.samples.append((img, 1))

        for img in fake.glob("*.jpg"):
            self.samples.append((img, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")

        image = self.transform(image)

        return image, label