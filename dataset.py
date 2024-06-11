from torch.utils.data import Dataset
import random
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
class NeedleDataset(Dataset):
    def __init__(
            self,
            meta_file,
            img_dir,
            mask_dir = None,
            mode = 'train',
            augment = True
    ):
        self.mode = mode
        self.augment = augment
        self.meta_info = pd.read_csv(meta_file)
        self.id_list = [file_name.split('.')[0] for file_name in os.listdir(img_dir)][:10]
        self.img_list = []
        self.class_list = []
        self.mask_list = []

        if self.mode == 'train':
            self.img_dir = img_dir
            self.mask_dir = mask_dir
            for id in tqdm(self.id_list):
                self.class_list.append(self.meta_info.loc[self.meta_info['imageID'] == int(id), 'status'].iloc[0])
                img = Image.open(os.path.join(self.img_dir, str(id) + '.jpg')).convert('L')
                self.img_list.append(transforms.ToTensor()(img))
                mask = Image.open(os.path.join(self.mask_dir, str(id) + '_mask.png')).convert('L')
                self.mask_list.append(transforms.ToTensor()(mask))
        elif self.mode == 'test':
            self.img_dir = img_dir
            for id in tqdm(self.id_list):
                img = Image.open(os.path.join(self.img_dir, str(id) + '.jpg')).convert('L')
                self.img_list.append(transforms.ToTensor()(img))


    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        if self.mode == 'train':
            img, mask, label = self.img_list[idx], self.mask_list[idx], self.class_list[idx]
            if self.augment:
                hflip = random.random() < 0.5
                vflip = random.random() < 0.5
                dflip = random.random() < 0.5

                def augment(x):
                    if hflip:
                        x = x.flip(-2)
                    if vflip:
                        x = x.flip(-1)
                    if dflip:
                        x = x.transpose(-2, -1)
                    return x

                img = augment(img)
                mask = augment(mask)
            return img, mask, label, self.id_list[idx]
        
        elif self.mode == 'test':
            img = self.img_list[idx]
            return img, self.id_list[idx]
    