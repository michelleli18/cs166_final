import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class ImageLoader(Dataset):
    def __init__(self, folder_path='ffhq_1k', transform=None):
        self.transform = transform
        self.image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            # image = np.transpose(image, (1, 2, 0))
        
        return image

# Perform transformations as described in [1] for comparitability
transform = transforms.Compose([
    transforms.Resize((256, 256)), # See [2]
    transforms.ToTensor(),  # Make into tensor, normalize to [0, 1] 
])
FFHQ_dataset = ImageLoader(transform=transform)
FFHQ_dataloader = DataLoader(FFHQ_dataset, batch_size=1, shuffle=False, num_workers=1)

# [1]: Chung et al DPS paper: https://dps2022.github.io/diffusion-posterior-sampling-page/
# [2]: Commonly used evaluation dataset FFHQ downsampled to 256 x 256, used in [1] https://www.kaggle.com/datasets/denislukovnikov/ffhq256-images-only/data. Last 10k images (out of 70k) of dataset are used for validation, we take the last 1k of these 10k validation images since [1] also only evaluated on 1k images.