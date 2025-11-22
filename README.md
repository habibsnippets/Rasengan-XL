# Rasengan-XL 

This project focuses on fine-tuning Stable Diffusion XL (SDXL) on a Naruto-style dataset to generate high quality anime-style images inspired by the Naruto universe.

The model was fine-tuned using multiple parameter-efficient fine-tuning (PEFT) strategies to study efficiency, memory behavior, and output quality trade-offs.

Training Approaches Used:

The model was trained using three different approaches.

Standard LoRA (No Quantization): 

Training: 

https://github.com/habibsnippets/Rasengan-XL/blob/master/Rasengan-XL-LoRA/Rasengan_XL_Training%20(1).ipynb


Inference:

https://github.com/habibsnippets/Rasengan-XL/blob/master/Rasengan-XL-LoRA/inference.ipynb


Results:

![Bill Gates](https://i.postimg.cc/3xMcPnyK/Screenshot-from-2025-11-22-17-20-21.png)

In this approach:

* The base SDXL model was loaded in full precision.
* LoRA adapters were injected into attention layers.
* No quantization was applied.

This served as the baseline fine-tuning method.

Key characteristics:

Base model kept in full precision.

Only LoRA adapters were trained.

Lower percentage of trainable parameters.


QLoRA Approaches:

Two different QLoRA configurations were explored using 4-bit NF4 quantization.

QLoRA quantizes the base weights to 4-bit and then trains low-rank adapters on top of them. This significantly reduces memory usage and allows training on limited hardware.

a) Aggressive QLoRA:

Training:

https://github.com/habibsnippets/Rasengan-XL/blob/master/Rasengan-XL-QLoRA/Rasengan_XL_Training_QLoRA_Aggressive.ipynb

Inference:

https://github.com/habibsnippets/Rasengan-XL/blob/master/Rasengan-XL-QLoRA/inference_qlora_aggressive.ipynb

Results:


![Bill gates](https://i.postimg.cc/LX47x5JR/Screenshot-from-2025-11-22-17-19-08.png)

This version was designed to maximize model adaptability

Characteristics:

Higher LoRA rank - 32

Larger number of target modules

Target modules included attention projections and feedforward layers (q, k, v, output, and FFNs).

Higher number of trainable parameters.

Higher representational capacity.


b) Normal QLoRA:

Training + Inferenece (Kaggle) - single notebook:

https://github.com/habibsnippets/Rasengan-XL/blob/master/Rasengan-XL-QLoRA/Rasengan-XL_QLoRA_Normal.ipynb


Results:

![Bill](https://i.postimg.cc/Pq6ss1sV/Screenshot-from-2025-11-22-17-21-32.png)

This version was designed to be more conservative and efficient.

Characteristics:

Lower LoRA rank - 16

Target modules limited to attention layers only (q, k, v, and output projections).


DoRA (Weight-Decomposed LoRA):

Training + Inference:

https://github.com/habibsnippets/Rasengan-XL/blob/master/Rasengan-XL_DORA.ipynb

Results:

![Bill](https://i.postimg.cc/prfsxt5L/Screenshot-from-2025-11-22-17-23-39.png)

DoRA was used as a third approach to test whether decoupling direction and magnitude improves fine-tuning quality.

In this approach:

Adapter weights were decomposed into magnitude and direction.

Base model weights remained frozen.

The fine-tuning focused on more stable weight updates.


Hardware Setup

The experiments were performed on limited VRAM environments such as:

NVIDIA Tesla T4 (16GB)

Kaggle GPUs

Various memory optimization strategies such as:

Gradient checkpointing

Mixed precision

4-bit quantization

VAE slicing

Attention slicing

were used to make training feasible.

Outputs

The final outcome includes:

Three fine-tuned SDXL variants (LoRA, QLoRA, DoRA).


Inference scripts for base vs fine-tuned comparison.

Side-by-side visual results for each approach.


Final Note:
> While Full-precision LoRA achieved the lowest final loss (0.0971), QLoRA with aggressive quantization reached a comparable loss of 0.0980 while significantly reducing memory requirements. Interestingly, DoRA achieved similar loss levels (~0.1) while maintaining step times comparable to Normal LoRA, making it a strong candidate for efficient large-scale fine-tuning when both speed and memory are constraints.
