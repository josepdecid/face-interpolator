from PIL import Image
from torch.utils.data import Dataset

from utils.system import join_path


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

        return image, attributes

    def get_image_attributes(self, root):
        attributes_dict = {}
        with open(join_path(root, 'Anno', 'list_attr_celeba.txt'), 'r') as f:
            f.readline()
            f.readline()
            for line in f.readlines():
                line_attributes = line.split()
                attributes_dict[line_attributes[0]] = line_attributes[1:]

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
