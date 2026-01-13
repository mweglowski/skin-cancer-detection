import torch
from torchvision import datasets, transforms

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Load dataset
    dataset = datasets.ImageFolder("data/train-image", transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    mean = torch.zeros(3, device=device)
    std = torch.zeros(3, device=device)
    nb_samples = 0

    for images, _ in loader:
        images = images.to(device)

        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)

        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print("Dataset mean:", mean.cpu())
    print("Dataset std:", std.cpu())
