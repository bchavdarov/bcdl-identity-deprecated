"""
Train BCSimpleCNN on bcdl-faces, then test it image by image.

Run from the project root (inside the jetson-inference container):
    python3 bcdl_train.py --epochs 30
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from bcdl_model import BCSimpleCNN, CLASSES
from bcdl_data import make_loaders


def evaluate(model, loader, device):
    """Accuracy in % over a whole loader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predicted = model(images).argmax(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def test_each_image(model, dataset, device):
    """One line per image: true person, predicted person, confidence, OK/ERR. Then the accuracy."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            image, label = dataset[i]
            name = dataset.samples[i][0]

            probs = torch.softmax(model(image.unsqueeze(0).to(device)), dim=1)[0]
            conf, pred = probs.max(0)
            pred = pred.item()

            ok = pred == label
            correct += ok
            status = 'OK ' if ok else 'ERR'
            print(f'{name}  true={CLASSES[label]:<15} pred={CLASSES[pred]:<15} '
                  f'conf={conf.item() * 100:5.1f}%  [{status}]')

    print(f'Accuracy: {correct}/{len(dataset)} = {100.0 * correct / len(dataset):.1f}%')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data-dir', type=str, default='data/bcdl-faces')
    parser.add_argument('--out', type=str, default='models/bcdl.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 1. Load the data
    train_loader, val_loader = make_loaders(args.data_dir, CLASSES, batch_size=args.batch_size)
    print(f'train images: {len(train_loader.dataset)}, validation images: {len(val_loader.dataset)}')

    model = BCSimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 2. Learn: passes over the training images
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        acc = evaluate(model, val_loader, device)
        print(f'Epoch {epoch:2d}/{args.epochs} | loss: {running_loss / len(train_loader):.4f} | '
              f'val acc: {acc:5.1f}% | time: {time.time() - t0:.1f}s')

        if acc >= best_acc:            # >= so a model is always saved, even at 0%
            best_acc = acc
            torch.save(model.state_dict(), args.out)
            print(f'  -> saved best model ({acc:.1f}%) to {args.out}')

    # 3. Test: reload the best model and check every validation image
    print(f'\nTesting best model ({best_acc:.1f}%):')
    model.load_state_dict(torch.load(args.out, map_location=device))
    test_each_image(model, val_loader.dataset, device)


if __name__ == '__main__':
    main()
