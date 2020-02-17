from torch import nn
from torchvision import models
import torch
import torch.nn.functional as F
import warnings


class Classifier(nn.Module):
    def __init__(self, inputs, hidden_one, hidden_two, outputs, p):
        super().__init__()
        self.fc1 = nn.Linear(inputs, hidden_one)
        self.fc2 = nn.Linear(hidden_one, hidden_two)
        self.fc3 = nn.Linear(hidden_two, outputs)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


def build_classifer(inputs=2208, hidden_one=512, hidden_two=256, outputs=102, p=0.2):
    return Classifier(inputs, hidden_one, hidden_two, outputs, p)


def build_model_from_state_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    # filter deprecated warnings (from pytorch download): https://stackoverflow.com/a/14463362/1113872
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pretrained = getattr(models, checkpoint['transfer_model_name'])(pretrained=True)
    classifier = build_classifer(
        checkpoint['inputs'],
        checkpoint['hidden_in'],
        checkpoint['hidden_out'],
        checkpoint['outputs']
    )
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    setattr(pretrained, 'classifier', classifier)
    pretrained.class_to_idx = checkpoint['class_to_idx']
    pretrained.eval()
    
    return pretrained, checkpoint


def predict(image, model, cat_to_name, topk, use_gpu):
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
    class_names = [cat_to_name[c] for c in classes]
    
    return class_names[0], probs[0], classes
    