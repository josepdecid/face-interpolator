from face_interpolator.data import CelebADataModule

mean = 0.0
std = 0.0
nb_samples = 0

celeba = CelebADataModule('datasets/CelebA')
celeba.setup()
loader = celeba.train_dataloader()

idx = 0
for data in loader:
    data = data[0]
    idx += 1
    print(idx, 'of', len(loader))
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print(mean)
print(std)
