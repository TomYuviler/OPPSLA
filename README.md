# One Pixel Adversarial Attacks via Sketched Programs
This repository contains the code for the paper "One Pixel Adversarial Attacks via Sketched Programs" by Tom Yuviler and Dana Drachsler-Cohen from the Technion, Israel. The paper proposes a novel approach to generate one pixel adversarial attacks with significantly fewer queries to the network by leveraging program synthesis.

## Abstract

Neural networks are successful in various tasks but are also susceptible to adversarial examples. An adversarial
example is generated by adding a small perturbation to a correctly-classified input with the goal of causing
a network classifier to misclassify. In one pixel attacks, an attacker aims to fool an image classifier by
modifying a single pixel. This setting is challenging for two reasons: the perturbation region is very small and
the perturbation is not differentiable. To cope, one pixel attacks iteratively generate candidate adversarial
examples and submit them to the network until finding a successful candidate. However, existing works
require a very large number of queries, which is infeasible in many practical settings, where the attacker
is limited to a few thousand queries to the network. We propose a novel approach for computing one pixel
attacks. The key idea is to leverage program synthesis and identify an expressive program sketch that enables
to compute adversarial examples using significantly fewer queries. We introduce OPPSLA, a synthesizer
that, given a classifier and a training set, instantiates the sketch with customized conditions over the input’s
pixels and the classifier’s output. OPPSLA employs a stochastic search, inspired by the Metropolis-Hastings
algorithm, that synthesizes typed expressions enabling minimization of the number of queries to the classifier.
We further show how to extend OPPSLA to compute few pixel attacks minimizing the number of perturbed
pixels. We evaluate OPPSLA on several deep networks for CIFAR-10 and ImageNet. We show that OPPSLA
obtains a state-of-the-art success rate, often with an order of magnitude fewer queries than existing attacks.
We further show that OPPSLA’s programs are transferable to other classifiers, unlike existing one pixel attacks,
which run from scratch on every classifier and input.


## Installation

1. Clone the repository:

git clone https://github.com/TomYuviler/OPPSLA.git


2. Navigate to the cloned repository and install the required dependencies:

cd OnePixelAttacks-SketchedPrograms
pip install -r requirements.txt

csharp
Copy code

## Usage

_TODO: Provide instructions on how to use the code, including input formats, example commands, and output descriptions._
The following are examples of how to use OPPSLA to synthesize one/few pixel adversarial programs, and how to use these programs to attack the model.
### Synthesize program and use it to perform one pixel attack on CIFAR-10 VGG-16BN model, on all possible ten classes 
```bash
python3 synthesize.py --model "vgg16" --data_set "cifar10" --num_train_images 50 --classes_list 0 1 2 3 4 5 6 7 8 9 --max_iter 210
```
```bash
python3 attack.py --model "vgg16" --data_set "cifar10" --max_queries 10000 --classes_list 0 1 2 3 4 5 6 7 8 9 --program_path "vgg16_cifar10.pkl" --max_k 1
```

### Synthesize program and use it to perform one pixel attack on CIFAR-10 ResNet18 model, on the five first classes with finer granularity of perturbations. 
```bash
python3 synthesize.py --model "resnet18" --data_set "cifar10" --num_train_images 50 --classes_list 0 1 2 3 4 --max_iter 210 --g 1 --max_g 5
```
```bash
python3 attack.py --model "resnet18" --data_set "cifar10" --max_queries 10000 --classes_list 0 1 2 3 4 --program_path "resnet18_cifar10.pkl" --max_k 1 --g 1 --max_g 5
```

python3 synthesize.py --model "vgg16" --data_set "cifar10" --num_train_images 50 --g 0 --max_g 0 --classes_list 0 1 2 3 4 5 6 7 8 9 --max_iter 210


## Results

_TODO: Summarize the results obtained from the experiments performed in the paper._

## Models
CIFAR-10 models adapted from [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.4431043.svg)](http://dx.doi.org/10.5281/zenodo.4431043). 

## Citation

If you find this work useful in your research, please cite:

@inproceedings{yuviler2023onepixel,
title={One Pixel Adversarial Attacks via Sketched Programs},
author={Tom Yuviler and Dana Drachsler-Cohen},
booktitle={Proceedings of the ...},
year={2023},
organization={...},
}


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more information.
