"""
Link between the training script and the identification script.

    training script                          identification script
    ---------------                          ---------------------
    model learns ...                         model = load_checkpoint(path)
    save_checkpoint(path, model, CLASSES)    model(image) -> index -> classes[index]
                     |                       ^
                     +--> models/bcdl.pth ---+   (one file)

The file holds everything the identification side needs to rebuild the
exact same network and to turn its output number back into a name.
"""

import os

import torch

from bcdl_model import BCSimpleCNN


def save_checkpoint(path, model, classes, epoch=None, val_acc=None):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),                 # the learned weights
        'conv1_out': model.features[0].out_channels,       # width of the network, needed to rebuild it
        'classes': list(classes),                          # index -> person name
        'epoch': epoch,                                    # for information only
        'val_acc': val_acc,                                # for information only
    }, path)


def load_checkpoint(path, device='cpu'):
    ckpt = torch.load(path, map_location=device)

    # 1. Build an empty network with the same shape as the saved one
    model = BCSimpleCNN(num_classes=len(ckpt['classes']), conv1_out=ckpt['conv1_out'])
    # 2. Fill it with the learned weights
    model.load_state_dict(ckpt['model_state'])
    # 3. Switch to identification mode: Dropout off, BatchNorm uses its stored statistics
    model.to(device).eval()

    return model, tuple(ckpt['classes']), ckpt


if __name__ == '__main__':
    # Self-test: save an untrained model, load it back, check both give identical outputs.
    from bcdl_model import CLASSES

    path = 'models/_selftest.pth'
    a = BCSimpleCNN(conv1_out=16).eval()
    save_checkpoint(path, a, CLASSES, epoch=0, val_acc=0.0)

    b, classes, info = load_checkpoint(path)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        same = torch.equal(a(x), b(x))

    print('classes:', classes)                                    # ('Al Pacino', 'Robert De Niro')
    print('conv1_out:', info['conv1_out'], 'epoch:', info['epoch'])  # 16 0
    print('identical outputs:', same)                             # True
    os.remove(path)
