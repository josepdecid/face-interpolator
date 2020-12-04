import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from face_interpolator.utils.constants import MEAN, STD
from face_interpolator.utils.system import join_path


class CelebaDataset(Dataset):
    """
    The CelebA dataset  is a large-scale face attributes dataset with more than 200K celebrity images, each with
    40 attribute annotations.
    A sample from this dataset will return the celebrity image and the list of attributes related to it.

    Args:
        - root (string): Root directory of dataset where the data is stored following the structure mentioned
            in the README.md file.
        - split (string, optional): It can take one of the following values:
            - 'train'(Default) = Creates dataset from training set.
            - 'val' = Creates dataset from validation set.
            - 'test' = Creates dataset from test set.
        - transformations (callable, optional): A function/transform that takes as input a PIL image
            and returns a transformed version.

    Attributes:
        - root (string): Where the root arg is stored.
        - split_images (list): List containing all the image file names from the chosen data split.
        - image_attributes_dict (dict): Dictionary containing the attributes values for each image,
            the keys are the image file names.
        - size (int): Length of the dataset for the chosen data split.
        - transform (callable): Where transform arg is stored.
    """
    image_attributes_size = 40
    attribute_names = ['5 o Clock Shadow', 'Arched Eyebrows', 'Attractive', 'Bags Under Eyes', 'Bald', 'Bangs',  # 0 - 5
                       'Big Lips', 'Big Nose', 'Black Hair', 'Blond Hair', 'Blurry', 'Brown Hair', 'Bushy Eyebrows',  # 6 - 12
                       'Chubby', 'Double Chin', 'Eyeglasses', 'Goatee', 'Gray Hair', 'Heavy Makeup', 'High Cheekbones',  # 13 - 19
                       'Male', 'Mouth Slightly Open', 'Mustache', 'Narrow Eyes', 'No Beard', 'Oval Face', 'Pale Skin',  # 20 - 26
                       'Pointy Nose', 'Receding Hairline', 'Rosy Cheeks', 'Sideburns', 'Smiling', 'Straight Hair',  # 27 - 32
                       'Wavy Hair', 'Wearing Earrings', 'Wearing Hat', 'Wearing Lipstick', 'Wearing Necklace',  # 33 - 37
                       'Wearing Necktie', 'Young']

    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split_images = self.get_images_from_data_split(root, split)
        self.image_attributes_dict = self.get_image_attributes(root)
        self.size = len(self.split_images)
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image = Image.open(join_path(self.root, 'Img', self.split_images[index]))
        attributes = self.image_attributes_dict[self.split_images[index]]

        if self.transform is not None:
            image = self.transform(image)
            attributes = torch.from_numpy(attributes)
            attributes[attributes == -1] = 0

        return image, attributes

    def get_image_attributes(self, root):
        attributes_dict = {}
        with open(join_path(root, 'Anno', 'list_attr_celeba.txt'), 'r') as f:
            f.readline()
            f.readline()
            for line in f.readlines():
                line_attributes = line.split()
                attributes_dict[line_attributes[0]] = np.array(line_attributes[1:]).astype(np.float32)

        return attributes_dict

    @staticmethod
    def get_images_from_data_split(root, split):
        split_index_dict = {'train': '0', 'val': '1', 'test': '2'}
        split_index = split_index_dict[split]
        file_names_list = []
        with open(join_path(root, 'Eval', 'list_eval_partition.txt'), 'r') as split_file:
            lines = split_file.readlines()
            for line in lines:
                words = line.split()
                if words[-1] == split_index:
                    file_names_list.append(words[0])

        return file_names_list


class CelebADataModule(pl.LightningDataModule):

    def __init__(self, dataset_root: str, batch_size: int = 64, num_workers: int = 0):
        super().__init__()
        self.dataset_root = dataset_root
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomPerspective(distortion_scale=0.05, p=0.2),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_set = CelebaDataset(self.dataset_root, split='train', transform=self.transform)
            self.val_set = CelebaDataset(self.dataset_root, split='val', transform=self.transform)

        if stage == 'test' or stage is None:
            self.test_set = CelebaDataset(self.dataset_root, split='test', transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
