# Evaluating the Impact of Data Augmentation on Image Classification

A reproducible, modular framework to compare the effect of various data augmentation techniques on image classification accuracy, reliability, and robustness using CNN, ResNet, and EfficientNet across CIFAR-10, MNIST, and ImageNet.

---

## 🚀 Getting Started

### 1. Clone the repository

```
git clone https://github.com/iAnisDev/data-augmentation-impact.git
cd data-augmentation-impact
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

## 📂 Directory Structure

```
data-augmentation-impact/
├── data/
├── src/
│   ├── main.py
│   └── ...
├── experiments/
├── results/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ⚡️ Usage

All tasks (downloading data, preprocessing, training, evaluation) can be managed via the unified CLI interface.

### Basic Command Format

```
python src/main.py [ACTION FLAGS] [OPTIONS]
```

#### Available Action Flags

| Flag            | Shortcut | Description                                                             |
|-----------------|----------|-------------------------------------------------------------------------|
| --load-data     | --ld     | Download datasets (CIFAR-10, MNIST, etc.)                               |
| --preprocess    | --pp     | Preprocess and clean data                                               |
| --train         | --tr     | Train models (choose model/dataset/aug)                                 |
| --evaluate      | --ev     | Evaluate models and output metrics/plots                                |
| --all           | --a      | Run the entire pipeline (load, preprocess, train, evaluate)             |
| --augment       | --aug    | Specify augmentation strategy (traditional, advanced, gan, etc.)        |
| --config        | -c       | Use experiment configuration file (YAML/JSON)                           |
| --help          | -h       | Show help message                                                       |

---

### 🏁 Example Workflows

**1. Download data only**

```
python src/main.py --load-data
```

**2. Preprocess data**

```
python src/main.py --preprocess --dataset cifar10
```

**3. Train a model (ResNet, with traditional augmentation, on CIFAR-10)**

```
python src/main.py --train --model resnet --dataset cifar10 --augment traditional
```

**4. Evaluate all models and output metrics/plots**

```
python src/main.py --evaluate
```

**5. Run the entire pipeline on MNIST with GAN augmentation**

```
python src/main.py --all --dataset mnist --model efficientnet --augment gan
```

**6. Use an experiment configuration file**

```
python src/main.py --all --config experiments/configs/baseline_resnet_cifar10.yaml
```

---

## 🛠️ Arguments Reference

| Argument       | Description                                                      | Default      |
|----------------|------------------------------------------------------------------|--------------|
| --model        | Model to train (`cnn`, `resnet`, `efficientnet`)                 | `cnn`        |
| --dataset      | Dataset to use (`cifar10`, `mnist`, `imagenet`)                  | `cifar10`    |
| --augment      | Augmentation method (`none`, `traditional`, `advanced`, `auto`, `gan`) | `none` |
| --epochs       | Number of training epochs                                        | `20`         |
| --batch-size   | Batch size                                                       | `64`         |
| --config       | Path to config file for advanced setups                          | *optional*   |

*See `python src/main.py --help` for the full list of options and descriptions.*

---

## 📊 Results

Outputs such as logs, metrics, and plots will be saved in the `results/` directory.

---


## 📄 License

MIT License

---

## 📚 References

* List of all academic papers and links to datasets, see `docs/references.md`.
