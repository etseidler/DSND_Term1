from PIL import Image
from torchvision import datasets, transforms
import torch
import numpy as np

def process_image(image):
    resize_num = 256
    w, h = image.size
    (new_w, new_h) = (1 + (resize_num * w // h), resize_num) if w > h else (resize_num, 1 + (resize_num * h // w))
    resized = image.resize((new_w, new_h))
    
    resized_w, resized_h = resized.size
    w_margin = (resized_w - 224) // 2
    h_margin = (resized_h - 224) // 2
    (left, upper, right, lower) = (w_margin, h_margin, resized_w - w_margin, resized_h - h_margin)
    cropped = resized.crop((left, upper, right, lower))
    
    np_image = np.array(cropped)
    np_converted = np_image / 255.0
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (np_converted - mean) / std
    
    reordered = np.transpose(normalized, (2, 0, 1))

    return reordered


def get_torched_image(image_path):
    img = Image.open(image_path)
    processed = process_image(img)
    # expand_dims used to account for model processing items in batches
    torched = torch.from_numpy(np.expand_dims(processed, axis=0))
    return torched


def create_transforms():
    resize_size = 255
    crop_size = 224
    norm_means = (0.485, 0.456, 0.406)
    norm_stds = (0.229, 0.224, 0.225)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_means, norm_stds)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_means, norm_stds)
        ]),
        'test': transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_means, norm_stds)
        ])
    }
    return data_transforms


def prep_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = create_transforms()
    
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }
    
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=True)
    }
    
    return dataloaders, image_datasets