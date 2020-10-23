from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder


def load_data(path, transforms, batch_size):
    dataset = ImageFolder(path, transforms)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    return data_loader
