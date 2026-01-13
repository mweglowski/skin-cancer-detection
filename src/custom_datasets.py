import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

import h5py
import io
from torch.utils.data import Dataset


class SkinLesionDataset(Dataset):
    def __init__(self, dataframe, root_dir, transforms=None):
        """
        Args:
            dataframe (pd.DataFrame): The dataframe object (train_df, val_df, etc.)
            root_dir (string): Directory with all the images (the flat folder).
            transform (callable, optional): Optional transform to be applied.
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # Use iloc to access by integer position, regardless of the DataFrame index
        row = self.dataframe.iloc[index]
        
        # Column 0 is the filename, Column 1 is the label
        # Adjust these keys if your CSV columns have specific names like 'image_id'
        img_name = os.path.join(self.root_dir, f'{row.iloc[0]}.jpg') 
        label = int(row.iloc[1])
        
        image = Image.open(img_name).convert('RGB')
        image = np.array(image)
        
        if self.transforms:
            image = self.transforms(image)
            
        return image, label



class SkinLesionHDF5Dataset(Dataset):
    def __init__(self, dataframe, hdf5_path, transforms=None):
        self.df = dataframe.reset_index(drop=True)
        self.hdf5_path = hdf5_path
        self.transforms = transforms
        self.h5_file = None

    def _open_hdf5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, "r")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        self._open_hdf5()

        image_id = self.df.iloc[idx, 0]
        label = int(self.df.iloc[idx, 1])

        raw_bytes = self.h5_file[image_id][()]
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        return image, label