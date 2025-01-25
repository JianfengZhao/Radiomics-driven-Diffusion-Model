import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader

class MedicalImageDataset(Dataset):
    def __init__(self, target_dir, condition_dir, mask_dir):
        self.target_dir = target_dir
        self.condition_dir = condition_dir
        self.mask_dir = mask_dir

        # Get the number of files in the target directory
        self.num_files = len([name for name in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, name))])

    def __len__(self):
        return self.num_files

    def preprocess_image(self, image):
        # Remove the first 15 slices and the last 12 slices
        image = image[:, :, 15:-12]
        # Normalize image to range [-1, 1]
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 2 - 1
        return image

    def __getitem__(self, idx):
        target_image_path = os.path.join(self.target_dir, f"{idx + 1}.nii.gz")
        condition_image_path = os.path.join(self.condition_dir, f"{idx + 1}.nii.gz")
        mask_image_path = os.path.join(self.mask_dir, f"{idx + 1}.nii.gz")

        target_image = nib.load(target_image_path).get_fdata()
        condition_image = nib.load(condition_image_path).get_fdata()
        mask_image = nib.load(mask_image_path).get_fdata()

        target_image = self.preprocess_image(target_image)
        condition_image = self.preprocess_image(condition_image)
        mask_image = self.preprocess_image(mask_image)

        return {
            'target': torch.tensor(target_image, dtype=torch.float32).unsqueeze(0),  # Add channel dimension
            'condition': torch.tensor(condition_image, dtype=torch.float32).unsqueeze(0),  # Add channel dimension
            'mask': torch.tensor(mask_image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        }

def get_dataloader(target_dir, condition_dir, mask_dir, batch_size, num_workers=0):
    dataset = MedicalImageDataset(target_dir, condition_dir, mask_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader