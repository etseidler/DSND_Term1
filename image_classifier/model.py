from torch import nn, optim
from torch.autograd import Variable
from torchvision import models
import torch
import torch.nn.functional as F
import warnings

    
class Classifier(nn.Module):
    def __init__(self, inputs, hidden_inputs, outputs, p):
        super().__init__()
        self.fc1 = nn.Linear(inputs, hidden_inputs)
        self.fc2 = nn.Linear(hidden_inputs, outputs)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

    
ALLOWED_ARCHS = [
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'densenet121',
    'densenet161',
    'densenet169',
    'densenet201'
]


def is_resnet(model_name):
    return model_name.startswith('resnet')


def build_classifier(inputs=2208, hidden=512, outputs=102, p=0.2):
    return Classifier(inputs, hidden, outputs, p)


def build_model_from_pretrained(model_name, hidden_layer_inputs, class_to_idx):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pretrained = getattr(models, model_name)(pretrained=True)
    classifier_inputs = list(pretrained.children())[-1].in_features
    classifier = build_classifier(classifier_inputs, hidden_layer_inputs)
    for param in pretrained.parameters():
        param.requires_grad = False
    pretrained.classifier = classifier
    if is_resnet(model_name):
        pretrained.fc = classifier
    pretrained.class_to_idx = class_to_idx
    return pretrained


def train(model, dataloaders, learning_rate, epochs, use_gpu):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    device = 'cuda' if use_gpu else 'cpu'
    model.to(device)
    
    for e in range(epochs):
        train_loss = 0
        for images, labels in dataloaders['train']:
            images, labels = images.to(device), labels.to(device)
            log_p = model(images)
            loss = criterion(log_p, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            optimizer.zero_grad()
        else:
            valid_loss = 0
            accuracy = 0
            with torch.no_grad():
                model.eval()
                for images, labels in dataloaders['valid']:
                    images, labels = images.to(device), labels.to(device)
                    log_p = model(images)
                    valid_loss += criterion(log_p, labels)
                    p = torch.exp(log_p)
                    _, top_class = p.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
        print(f'Epoch: {e+1}/{epochs}')
        print(f'Training loss: {round(train_loss/len(dataloaders["train"]), 3)}')
        print(f'Validation loss: {round(float(valid_loss/len(dataloaders["valid"])), 3)}')
        print(f'Validation accuracy: {round(float(100*accuracy/len(dataloaders["valid"])), 2)}%')
        print('-----------------------------------------')
        
        model.train()
    return model, optimizer


def save_checkpoint(model, epochs, optimizer, model_name, learning_rate, save_dir):
    checkpoint = {
        'epochs': epochs,
        'optimizer_state_dict': optimizer.state_dict(),
        'transfer_model_name': model_name,
        'class_to_idx': model.class_to_idx,
        'inputs': model.classifier.fc1.in_features,
        'hidden': model.classifier.fc1.out_features,
        'outputs': model.classifier.fc2.out_features,
        'learning_rate': learning_rate
    }
    if is_resnet(model_name):
        checkpoint['classifier_state_dict'] = model.fc.state_dict()
    else:
        checkpoint['classifier_state_dict'] = model.classifier.state_dict()
    
    checkpoint_loc = save_dir + '/checkpoint.pth'
    torch.save(checkpoint, checkpoint_loc)
    print(f'Checkpoint saved at {checkpoint_loc}')


def build_model_from_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    model_name = checkpoint['transfer_model_name']
    # filter deprecated warnings (from pytorch download): https://stackoverflow.com/a/14463362/1113872
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pretrained = getattr(models, model_name)(pretrained=True)
    classifier = build_classifier(
        checkpoint['inputs'],
        checkpoint['hidden'],
        checkpoint['outputs']
    )
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    if is_resnet(model_name):
        setattr(pretrained, 'fc', classifier)
    else:
        setattr(pretrained, 'classifier', classifier)
    pretrained.class_to_idx = checkpoint['class_to_idx']
    pretrained.eval()
    
    return pretrained, checkpoint


def predict(image, model, topk, use_gpu):
    device = 'cuda' if use_gpu else 'cpu'
    image = image.to(device)
    image = image.to(dtype=torch.float32)
    model.to(device)
    
    p = torch.exp(model(image))
    probs, cls_idx = p.topk(topk, sorted=True)
    if use_gpu:
        probs = probs.cpu()
        cls_idx = cls_idx.cpu()
    
    # dictionary inversion from https://stackoverflow.com/a/47312474/1113872
    idx_to_classes = {value: key for key, value in model.class_to_idx.items()}
    
    probs, classes = probs.detach().numpy()[0], [idx_to_classes[c] for c in cls_idx.detach().numpy()[0]]
    
    return classes, probs
