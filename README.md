# bcdl-identity

Experimental PyTorch CNN that recognizes a person's identity from a small set of
face images. It is being built step by step and targets the **NVIDIA Jetson Nano**.
Part of PhD research in computer vision.

> **Status:** early stage. The repository has just been created, and the code will
> be added in small steps. Sections marked *(planned)* describe the intended
> design, not working code.

## Goals

- Train a small custom CNN (not a pretrained backbone) that tells apart a handful
  of identities from few images per person.
- Keep the model light enough to train and run **on the device**.
- Deploy through ONNX → TensorRT and measure accuracy and latency on the Nano.

## Target environment

| Component      | Version                                  |
|----------------|------------------------------------------|
| Hardware       | Jetson Nano                              |
| JetPack / L4T  | 4.6.x                                    |
| PyTorch        | 1.10.0 (from the jetson-inference image) |
| Python         | 3.6                                      |
| ONNX opset     | 11 (supported by TensorRT 8.0 on JP 4.6) |
| Runtime        | [`dusty-nv/jetson-inference`][ji] Docker container |

[ji]: https://github.com/dusty-nv/jetson-inference

Start the container with this repository mounted:

```bash
git clone --recursive https://github.com/dusty-nv/jetson-inference
cd jetson-inference
docker/run.sh --volume /path/to/bcdl-identity:/bcdl-identity
cd /bcdl-identity
```

For the best results, set the Nano to maximum performance before training or
benchmarking:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

## Model: `BCSimpleCNN` *(planned)*

A compact VGG-style network, about 665K parameters with the default width:

```
Input 3×H×W
├─ [Conv3×3 → BN → ReLU] ×2  (c)     → MaxPool 2
├─ [Conv3×3 → BN → ReLU] ×2  (2c)    → MaxPool 2
├─ [Conv3×3 → BN → ReLU]     (4c)    → MaxPool 2
├─ AdaptiveAvgPool → 4×4              (input size independent: 32×32, 224×224, …)
└─ Flatten → FC 256 → ReLU → Dropout 0.4 → FC num_classes
```

`c` (`conv1_out`) defaults to 32. The first experiment uses two identities.

## Pipeline *(planned)*

The same workflow was already proven on a CIFAR-10 prototype (85.7% test
accuracy, trained on the Nano):

1. **Dataset:** `data/bcdl-faces/` holds 60 face images (`001.jpeg` …),
   30 each of Al Pacino and Robert De Niro. Labels are in
   `data/bcdl-faces/annotations.csv` (columns `imagename,person`).
2. **Training:** PyTorch with light augmentation; saves the best checkpoint as
   `.pth` under `models/`. Both `data/` and `models/` are git-ignored.
3. **Evaluation:** accuracy on the validation set and predictions on single images.
4. **Export:** `torch.onnx.export` with opset 11 and a dynamic batch axis.
5. **TensorRT:** build an engine with `trtexec` (FP16) and run inference from
   Python with `tensorrt` + `pycuda`.
6. **Benchmark:** compare PyTorch and TensorRT latency and accuracy on the device.

Example engine build:

```bash
/usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
```

## Repository layout

```
bcdl-identity/
└── README.md
```

This section will be updated as scripts are added.

## License

- **Dataset (bcdl-faces):** see [LICENSE.md](LICENSE.md). The annotations are
  CC BY 4.0. The images belong to their copyright holders and may be used only
  for non-commercial research.
- **Code:** not yet specified. All rights reserved by the author until a code
  license is added.
