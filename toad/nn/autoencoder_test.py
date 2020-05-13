from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from autoencoder import AutoEncoder

dataset = datasets.MNIST(
    root = './mnist',
    train = True,
    transform = transforms.ToTensor(),
    download = True,
)


loader = DataLoader(
    dataset,
    batch_size = 128,
    shuffle = True,
)

print(len(dataset))

ae = AutoEncoder(784, 200, 10)
ae.fit(loader)