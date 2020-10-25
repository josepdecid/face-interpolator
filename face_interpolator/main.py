from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from face_interpolator.data.celeba_dataset import CelebaDataset

if __name__ == '__main__':
    # TODO: Define config file
    batch_size = 32
    num_workers = 0

    transform = transforms.Compose([transforms.ToTensor()])

    train_set = CelebaDataset('../datasets/CelebA', split='train', transform=transform)
    val_set = CelebaDataset('../datasets/CelebA', split='val', transform=transform)
    test_set = CelebaDataset('../datasets/CelebA', split='test', transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # TODO: Remove lines below, used to show how the dataset works
    image, attributes = train_set[0]
    print(attributes)
    pil_img = transforms.ToPILImage()(image).convert("RGB")
    pil_img.show()
