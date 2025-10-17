import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch

normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
data_path = './datasets/cub200_cropped/'
train_dir = data_path + 'train_cropped/'
test_dir = data_path + 'test_cropped/'
train_push_dir = data_path + 'train_cropped/'

train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True,
    num_workers=2, pin_memory=False)

train_push_dataset = datasets.ImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
    ]))
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=128, shuffle=False,
    num_workers=2, pin_memory=False)

test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=128, shuffle=False,
    num_workers=2, pin_memory=False)

