import copy
import os
import time
import pandas as pd
from PIL import Image
import torch.utils.data
import numpy as np

class DeepFakeDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, train_file, batch_size=1, transform=None, train=True, image_type='fullface',
                 dataset='all', shuffle=False, images_per_folder=16, image_name_contains='',
                 image_doesnt_contain='', labels_repeat=False, max_len=None):

        self.labels_repeat = labels_repeat
        self.max_len = max_len
        self.root_dir = root_dir
        self.image_name_contains = image_name_contains
        self.image_doesnt_contain = image_doesnt_contain
        self.images_per_folder = images_per_folder
        self.train = train
        self.train_file_path = train_file
        self.transform = transform
        self.batch_size = batch_size
        self.image_type = image_type
        self.dataset = dataset
        self.label_df = pd.read_excel(os.path.join(self.root_dir, self.train_file_path))
        if self.train:
            self.label_df = self.label_df[self.label_df['TrainTest'] == 0].reset_index(drop=True)
            self.label_df = self.label_df.iloc[0:int(len(self.label_df) / self.batch_size) * self.batch_size] # to have complete batches
        else:
            self.label_df = self.label_df[self.label_df['TrainTest'] == 1].reset_index(drop=True)
            self.label_df = self.label_df.iloc[0:int(len(self.label_df) / self.batch_size) * self.batch_size] # to have complete batches

        if dataset != 'all':
            self.label_df = self.label_df[self.label_df['Dataset'] == dataset].reset_index(drop=True)
            self.label_df = self.label_df.iloc[0:int(len(self.label_df) / self.batch_size) * self.batch_size] # to have complete batches

        #self.label_df_original = copy.deepcopy(self.label_df)
        self.classes = list(self.label_df['ClassId'].unique())

        # deleting empty folders
        self._check_folders()

        self.len_label_df = len(self.label_df)
        if shuffle:
            self.shuffle()

        self.label_df_original = copy.deepcopy(self.label_df)


    def _check_folders(self):

        print('Checking folders...')
        folders = [os.path.join(self.root_dir, x) for x in
                   self.label_df.loc[:, 'VideoName'].apply(
                       lambda x: x.split('.')[0])]

        empty_folders = []
        for i in range(len(folders)):
            try:
                if len(os.listdir(folders[i])) < 100:
                    empty_folders.append(i)
            except:
                empty_folders.append(i)

        self.label_df = self.label_df.drop(index=empty_folders)

    def __getitem__(self, idx, image_dim=(3, 299, 299)):
        """Return (image, target) after resize and preprocessing."""

        folders = [os.path.join(self.root_dir, x) for x in
                               self.label_df.loc[idx * self.batch_size:(idx + 1) * self.batch_size - 1, 'VideoName'].apply(lambda x: x.split('.')[0])]
        if self.batch_size == -1:
            folders = [os.path.join(self.root_dir, x) for x in
                       self.label_df.loc[:, 'VideoName'].apply(
                           lambda x: x.split('.')[0])]

        data = []
        data_folder = []
        unread_imgs = []
        i = 0
        for folder in folders:

            empty_folder = False
            if os.path.isdir(folder):
                if self.image_name_contains == '':
                    images = [x for x in os.listdir(folder) if self.image_type in x][1:self.images_per_folder+1]
                elif self.image_doesnt_contain == '':
                    images = [x for x in os.listdir(folder) if self.image_type in x and self.image_name_contains in x][0:self.images_per_folder]
                else:
                    images = [x for x in os.listdir(folder) if self.image_type in x and self.image_name_contains in x and self.image_doesnt_contain not in x][
                             0:self.images_per_folder]
            else:
                unread_imgs.append(i)
                continue

            for image in images:

                img = os.path.join(folder, image)
                X = np.array(Image.open(img), dtype=np.float32)
                if np.max(X) > 1:
                    X = X / 255

                if self.transform:
                    X = self.transform(X)

                if X is not None:
                    #data.append(X)
                    data_folder.append(X)
                else:
                    unread_imgs.append(i)

                i += 1

            data.append(data_folder)
            data_folder = []

        labels = self.label_df.loc[idx * self.batch_size:(idx + 1) * self.batch_size - 1, 'ClassId'].values
        if self.batch_size == -1:
            labels = self.label_df.loc[:, 'ClassId'].values

        if unread_imgs:
            labels_temp = copy.copy(labels)
            labels = []
            for i in range(len(labels_temp)):
                if i not in unread_imgs:
                    labels.append(labels_temp[i])

        labels = torch.Tensor(labels)
        if self.batch_size == -1:
            return data, labels

        if self.batch_size == 1:
            data = torch.stack(data[0])
        else:
            if len(data) == 0:
                return None, None
            for st in data:
                if len(st) == 0:
                    return None, None
            stacks = [torch.stack(x) for x in data if len(x) == self.images_per_folder]
            if len(stacks) < len(labels):
                return None, None
            try:
                data = torch.stack(stacks)
            except:
                return None, None

        if self.labels_repeat:
            labels = labels.repeat_interleave(self.images_per_folder)
            data = data.reshape(-1, 3, 299, 299)

        return data, labels.type(torch.cuda.DoubleTensor)

    def __len__(self):
        """Returns the length of the dataset."""
        if self.max_len is not None:
            return self.max_len
        else:
            return int(len(self.label_df) / self.batch_size) - 1

    def shuffle(self, random_state=1):
        self.label_df = self.label_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    def augment_dataset(self, less_class=0):

        big_class = 1 if less_class == 0 else 0
        lesser_class = self.label_df_original[self.label_df_original['ClassId'] == less_class]
        bigger_class = self.label_df_original[self.label_df_original['ClassId'] == big_class]

        min_bigger_class = np.min([len(lesser_class) * 7, len(bigger_class)])
        self.label_df = pd.concat([lesser_class] * 3 + [bigger_class.sample(frac=1).iloc[0:min_bigger_class]],
                                  ignore_index=True).sample(frac=1).reset_index(drop=True)

        print(len(lesser_class) * 3 , 'Real Samples', min_bigger_class, 'Fake Samples')






