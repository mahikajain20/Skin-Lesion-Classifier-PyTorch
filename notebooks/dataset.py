
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import random


#Data preparation
class DDI_Dataset(Dataset):
    def __init__(self, root, csv_path=None, transform=None):
        if csv_path is None:
            csv_path = os.path.join(root, "ddi_metadata.csv")
        assert os.path.exists(csv_path), f"Path not found <{csv_path}>."
        self.root = root
        self.transform = transform
        self.annotations = pd.read_csv(csv_path)

        def is_malignant(x):
            return x == 1
        
        m_key = 'malignant'
        if m_key not in self.annotations:
            #no  lambda function
            self.annotations[m_key] = self.annotations['malignancy(malig=1)'].apply(is_malignant)
        self.annotations.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        img_path = os.path.join(self.root, row['DDI_file'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        target = int(row['malignant']) # 1 if malignant, 0 if benign
        skin_tone = row['skin_tone'] # Fitzpatrick- 12, 34, or 56
        return image, target, skin_tone

    def get_sample_image(self, index=None):
        if index is None:
            index = random.randint(0, len(self) - 1)
        row = self.annotations.iloc[index]
        img_path = os.path.join(self.root, row['DDI_file'])
        image = Image.open(img_path).convert('RGB')
        target = int(row['malignant'])
        skin_tone = row['skin_tone']
        return image, target, skin_tone
