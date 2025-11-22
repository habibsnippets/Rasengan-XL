# Technical Report: Comparative Analysis of QLoRA Fine-Tuning Strategies for SDXL

## 1. Executive Summary
This report details the technical approach used to fine-tune the Stable Diffusion XL (SDXL) Base 1.0 model to adopt the "Naruto" artistic style under strict hardware constraints (Tesla T4 GPU, 16GB VRAM). Two distinct configurations of Quantized Low-Rank Adaptation (QLoRA) were implemented and analyzed: "QLoRA-aggressive" and "QLoRA-normal."

Both approaches successfully utilized 4-bit quantization to fit the  model into memory. However, they differ significantly in their adaptation scope, trainable parameter count, and theoretical capacity for style retention. This report compares these strategies to justify the final model selection.

## 2. Common Optimization Architecture
To address the core challenge of memory optimization, both configurations share an identical foundation designed to minimize VRAM usage without sacrificing training stability.

**Base Model Loading**
The SDXL Base 1.0 model was loaded using `bitsandbytes` 4-bit Normal Float (`nf4`) quantization. This technique compresses the model weights, reducing the VRAM footprint as well.

**Training Optimizations**
Both configurations employed the following techniques:
* **Gradient Checkpointing:** This was enabled to trade compute speed for memory. Instead of storing all intermediate activations during the forward pass, the model re-computes them during the backward pass, significantly lowering peak memory usage.
* **8-bit Adam Optimizer:** The `AdamW8bit` optimizer was utilized to reduce the memory required for optimizer states by approximately 75% compared to standard 32-bit AdamW.
* **Mixed Precision:** Training was conducted in FP16 mixed precision to further reduce memory bandwidth pressure and storage requirements.

## 3. Configuration Analysis

### Strategy A: QLoRA-aggressive 
This configuration prioritized maximum learning capacity to ensure the distinct "Naruto" art style—which involves complex changes in shading, line work, and color palettes—was accurately captured.

* **Rank (r):** Set to 32. A higher rank allows the low-rank matrices to capture more complex correlations and features in the update steps.
* **Target Modules:** This strategy targeted a comprehensive set of linear layers, including both the Self-Attention blocks (`to_q`, `to_k`, `to_v`, `to_out.0`) and the Feed-Forward Networks or MLPs (`add_k_proj`, `add_v_proj`, `ff.net.0.proj`, `ff.net.2`).
* **Trainable Parameters:** This resulted in approximately 83.7 million trainable parameters, representing roughly 5.45% of the model's total parameters.

**Motivation:**
The Feed-Forward Networks (FFNs) in Transformer architectures are often hypothesized to store knowledge and specific concepts. By targeting these layers in addition to attention mechanisms, QLoRA-aggressive allows the model to overwrite generic image concepts with specific anime-style representations effectively.

### Strategy B: QLoRA-normal 
This configuration was designed as a resource-minimalist fallback to guarantee training stability if the aggressive strategy exceeded memory limits.

* **Rank (r):** Set to 16. 
* **Target Modules:** This strategy targeted *only* the Self-Attention mechanisms (`to_q`, `to_k`, `to_v`, `to_out.0`), leaving the MLP layers frozen.
* **Trainable Parameters:** This resulted in approximately 23.2 million trainable parameters, representing roughly 1.57% of the model's total parameters.

**Motivation:**
Targeting only attention layers effectively changes how the model relates different tokens to each other (composition and layout) but has less capacity to alter the fundamental textures and rendering style stored in the FFNs. This makes it safer for hardware but potentially less effective for heavy style transfer.

## 4. Impact Assessment

**Memory Consumption**
QLoRA-aggressive consumes higher VRAM due to the increased number of gradients and optimizer states required for 83 million parameters compared to 23 million. However, testing confirmed that even the aggressive strategy remained within the 16GB VRAM limit of the Tesla T4 when combined with gradient checkpointing and a batch size of 1.

**Style Capture Capabilities**
The disparity in parameter count (83.7M vs 23.2M) indicates a massive difference in expressivity. QLoRA-normal risks under-fitting the style, potentially producing images that look like "SDXL attempting anime" rather than authentic "Naruto style." QLoRA-aggressive provides the necessary capacity to fundamentally shift the model's output distribution toward the target domain.

