from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def lsun_loader(path, batch_size):
    def loader(transform):
        data = datasets.LSUNClass(
            path, transform=transform,
            target_transform=lambda x: 0)
        data_loader = DataLoader(
            data, shuffle=False, batch_size=batch_size, num_workers=8)

        return data_loader

    return loader


def sample_data(dataset, batch_size, image_size=4):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset.transform = transform
    loader = DataLoader(dataset, shuffle=True,
                        batch_size=batch_size, num_workers=16)

    return loader
