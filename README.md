One Pixel Adversarial Attacks via Sketched Programs
This repository contains the code for the paper "One Pixel Adversarial Attacks via Sketched Programs" by Tom Yuviler and Dana Drachsler-Cohen from the Technion, Israel. The paper proposes a novel approach to generate one pixel adversarial attacks with significantly fewer queries to the network by leveraging program synthesis.

Abstract
Neural networks are successful in various tasks but are also susceptible to adversarial examples. An adversarial example is generated by adding a small perturbation to a correctly-classified input with the goal of causing a network classifier to misclassify. In one pixel attacks, an attacker aims to fool an image classifier by modifying a single pixel. This setting is challenging for two reasons: the perturbation region is very small and the perturbation is not differentiable.

Existing works on one pixel attacks require a very large number of queries, which is infeasible in many practical settings where the attacker is limited to a few thousand queries to the network. We propose a novel approach for computing one pixel attacks by leveraging program synthesis and identifying an expressive program sketch that enables computing adversarial examples using significantly fewer queries.

We introduce OPPSLA, a synthesizer that, given a classifier and a training set, instantiates the sketch with customized conditions over the input’s pixels and the classifier’s output. OPPSLA employs a stochastic search, inspired by the Metropolis-Hastings algorithm, that synthesizes typed expressions enabling minimization of the number of queries to the classifier. We further show how to extend OPPSLA to compute few pixel attacks minimizing the number of perturbed pixels.

We evaluate OPPSLA on several deep networks for CIFAR-10 and ImageNet. We show that OPPSLA obtains a state-of-the-art success rate, often with an order of magnitude fewer queries than existing attacks. We further show that OPPSLA’s programs are transferable to other classifiers, unlike existing one pixel attacks, which run from scratch on every classifier and input.

Installation
Clone the repository:
bash
Copy code
git clone https://github.com/username/OnePixelAttacks-SketchedPrograms.git
Navigate to the cloned repository and install the required dependencies:
bash
Copy code
cd OnePixelAttacks-SketchedPrograms
pip install -r requirements.txt
Usage
TODO: Provide instructions on how to use the code, including input formats, example commands, and output descriptions.

Results
TODO: Summarize the results obtained from the experiments performed in the paper.

Citation
If you find this work useful in your research, please cite:

less
Copy code
@inproceedings{yuviler2023onepixel,
  title={One Pixel Adversarial Attacks via Sketched Programs},
  author={Tom Yuviler and Dana Drachsler-Cohen},
  booktitle={Proceedings of the ...},
  year={2023},
  organization={...},
}
License
This project is licensed under the MIT License. See LICENSE for more information.
