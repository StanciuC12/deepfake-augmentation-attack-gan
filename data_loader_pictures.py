import copy
import os
import time
import pandas as pd
from PIL import Image
import torch.utils.data
import numpy as np

class DeepFakeDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, train_file, batch_size=1, transform=None, train=True, image_type='fullface',
                 dataset='all', nr_images_per_folder=5, reset_index=False, classes_used=None):

        self.root_dir = root_dir
        self.train = train
        self.train_file_path = train_file
        self.transform = transform
        self.batch_size = batch_size
        self.image_type = image_type
        self.dataset = dataset
        self.nr_images_per_folder = nr_images_per_folder
        self.label_df = pd.read_excel(os.path.join(self.root_dir, self.train_file_path))
        if self.train != 'all':
            if self.train:
                self.label_df = self.label_df[self.label_df['TrainTest'] == 0].reset_index(drop=True)
                self.label_df = self.label_df.iloc[0:int(len(self.label_df) / self.batch_size) * self.batch_size] # to have complete batches
            else:
                self.label_df = self.label_df[self.label_df['TrainTest'] == 1].reset_index(drop=True)
                self.label_df = self.label_df.iloc[0:int(len(self.label_df) / self.batch_size) * self.batch_size] # to have complete batches

        if dataset != 'all':
            self.label_df = self.label_df[self.label_df['Dataset'] == dataset].reset_index(drop=True)
            self.label_df = self.label_df.iloc[0:int(len(self.label_df) / self.batch_size) * self.batch_size] # to have complete batches

        if classes_used != None:
            self.label_df = self.label_df.loc[self.label_df['ClassId'].apply(lambda x: x in classes_used), :]

        #self.label_df_original = copy.deepcopy(self.label_df)
        self.classes = list(self.label_df['ClassId'].unique())

        self.len_label_df = len(self.label_df)
        # Repeating each folder
        self.label_df = pd.concat([self.label_df] * nr_images_per_folder)
        self.shuffle(state=42)

        self.label_df_original = copy.deepcopy(self.label_df)

        if reset_index:
            self.label_df = self.label_df.reset_index()

    def __getitem__(self, idx, image_dim=(3, 299, 299)):
        """Return (image, target) after resize and preprocessing."""

        folders = [os.path.join(self.root_dir, x) for x in
                               self.label_df.loc[idx * self.batch_size:(idx + 1) * self.batch_size - 1, 'VideoName'].apply(lambda x: x.split('.')[0])]

        data = []
        unread_imgs = []
        i = 0
        for folder in folders:

            empty_folder = False
            if os.path.isdir(folder):
                if self.image_type:
                    images = [x for x in os.listdir(folder) if self.image_type in x]
                else:
                    images = os.listdir(folder)
                image = images[np.random.randint(0, len(images) - 1)]
            else:
                unread_imgs.append(i)
                continue

            img = os.path.join(folder, image)
            X = np.array(Image.open(img), dtype=np.float32)
            if np.max(X) > 1:
                X = X / 255

            if self.transform:
                X = self.transform(X)

            if X is not None:
                data.append(X)
            else:
                unread_imgs.append(i)

            i += 1

        labels = self.label_df.loc[idx * self.batch_size:(idx + 1) * self.batch_size - 1, 'ClassId'].values

        if unread_imgs:
            labels_temp = copy.copy(labels)
            labels = []
            for i in range(len(labels_temp)):
                if i not in unread_imgs:
                    labels.append(labels_temp[i])

        labels = torch.Tensor(labels)
        data = torch.stack(data)

        return data, labels

    def __len__(self):
        """Returns the length of the dataset."""
        return int(len(self.label_df) / self.batch_size)

    def shuffle(self, state=None):
        self.label_df = self.label_df.sample(frac=1, random_state=state).reset_index(drop=True)

    def augment_dataset(self, less_class=0):

        big_class = 1 if less_class == 0 else 0
        lesser_class = self.label_df_original[self.label_df_original['ClassId'] == less_class]
        bigger_class = self.label_df_original[self.label_df_original['ClassId'] == big_class]

        min_bigger_class = np.min([len(lesser_class) * 7, len(bigger_class)])
        self.label_df = pd.concat([lesser_class] * 3 + [bigger_class.sample(frac=1).iloc[0:min_bigger_class]],
                                  ignore_index=True).sample(frac=1).reset_index(drop=True)

        print(len(lesser_class) * 3, 'Real Samples', min_bigger_class, 'Fake Samples')






