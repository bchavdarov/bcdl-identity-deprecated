"""
Custom CNN for recognition of identities, designed for Jetson Nano (JetPack 4.6.1, PyTorch 1.10.0).

Run inside the jetson-inference container:
    python3 simple-cnn.py ..............  ----- to be adjusted later on finished code

"""

import torch
import torch.nn as nn

CLASSES = ('Al Pacino', 'Robert De Niro')
NUM_CLASSES = len(CLASSES)

class BCSimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, conv1_out=32):
        super().__init__()
        c1, c2, c3 = conv1_out, conv1_out * 2, conv1_out * 4

        self.features = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(),
            nn.Conv2d(c1, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(),
            nn.Conv2d(c2, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4, 4)),  # works for 32x32, 224x224, etc.
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

if __name__ == "__main__":
    model = BCSimpleCNN()
    model.eval()
    out = model(torch.randn(2, 3, 224, 224))
    print(out.shape)  # torch.Size([2, 2])
    print(sum(p.numel() for p in model.parameters()))  # ~665k

