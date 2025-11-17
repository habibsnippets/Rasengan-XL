# Rasengan-XL - Technical Report

Fine Tune Model - GDrive - https://drive.google.com/drive/folders/1Ns17jEBQJu7kL5i7-9HUvAvgQRJjpmJj?usp=sharing

# Rasengan-XL

This technical report documents how I fine-tuned the Stable Diffusion XL (SDXL) 1.0 model on an NVIDIA Tesla T4 GPU (16 GB VRAM) in Google Colab to generate Naruto-style art. The goal was to adapt the 2.6-billion-parameter SDXL base model to a specific anime style using the lambdalabs/naruto-blip-captions dataset (1,221 image-caption pairs (https://huggingface.co/datasets/lambdalabs/naruto-blip-captions). Achieving this on a 16 GB GPU required a suite of memory-saving tricks, from algorithmic techniques like LoRA and 8-bit optimization to engineering tweaks like VAE slicing and CPU offloading. The following sections detail the end-to-end pipeline (training and inference) and the rationale behind each design choice.



## Training Methodology

For efficient training, I applied **Parameter-Efficient Fine-Tuning (PEFT)** with Low-Rank Adaptation (LoRA). In practice, I froze the original UNet weights and injected small trainable adapters into key attention layers (https://huggingface.co/docs/diffusers/main/en/training/lora). This means I only updated new low-rank matrices (`rank=16`, `alpha=32`) in the UNet’s self-attention and cross-attention projections (specifically the `"to_q"`, `"to_k"`, `"to_v"`, and `"to_out.0"` modules). In effect, only the LoRA layers (≈23 million parameters) were trainable — about 0.9% of the full model — drastically reducing VRAM for gradients and optimizer state.



- **Base Model:** Stable Diffusion XL 1.0 (SDXL) with ~2.6B parameters (StabilityAI’s SDXL Base 1.0).
- **Dataset:** `lambdalabs/naruto-blip-captions` (1,221 image-text pairs of Naruto scenes focusing the model on the Naruto anime style.
- **Resolution:** 512×512 pixels (downsized to fit T4 memory).
- **Precision:** Mixed precision training was used: UNet and text encoders in 16-bit, but *VAE encoder/decoder in 32-bit* to ensure numerical stability.

**LoRA Configuration:** The adapters were inserted via Hugging Face PEFT. Key settings were:



- **Rank (r):** 16. This is the inner dimension of each adapter’s low-rank matrices (higher rank = more capacity).
- **Alpha (α):** 32. A scaling factor for the LoRA weights.
- **Target Modules:** `["to_q", "to_k", "to_v", "to_out.0"]`. These are the projection layers of the UNet’s attention blocks where the LoRA adapters were applied.
- **Trainable Params:** Only ~23M new parameters (about 0.9% of 2.6B) were trained. The rest of the model remained frozen.

## Training Loop Implementation

The training loop was carefully structured to minimize memory use. Key steps:



- **Data Loading:** Images were preprocessed on the CPU: resized to 512×512, center-cropped, normalized, and batched. Text prompts were tokenized in CPU RAM as well to avoid unnecessary GPU memory usage.
- **VAE Encoding (float32):** Each image batch was encoded into latent space by the VAE in full precision on GPU. Using 32-bit here prevented NaN/overflow issues (see next section). After encoding, the latent tensors were cast to 16-bit for the diffusion step.
- **Diffusion (UNet + LoRA):** I added noise to the VAE latents and ran the (now LoRA-augmented) UNet to predict the noise residual. Since only the LoRA adapters were trainable, backpropagation stored gradients just for those small matrices. The rest of the UNet was frozen.
- **Optimization:** Only LoRA parameters were passed to the optimizer. I used 8-bit AdamW (via *bitsandbytes*) to update them. Switching to 8-bit AdamW offers roughly 75% memory savings over a standard 32-bit AdamW (https://huggingface.co/docs/bitsandbytes/v0.43.0/en/optimizers), which was crucial for fitting on 16 GB. For example, 8-bit AdamW uses 8-bit states instead of full 32-bit, drastically cutting optimizer overhead. The rest of the training (gradient clipping, scheduling) followed typical guidelines.

## Engineering Design Choices & Justifications

Each component of the pipeline was chosen to work around the T4’s memory limits:



### The VAE Precision Dilemma (Why Float32?)

Early in training, I encountered NaNs originating from the VAE’s encoding step. In the initial runs I had encoded images with the VAE in float16 on GPU, which caused numerical overflow. After investigating, I found others had reported SDXL’s VAE is not stable in 16-bit. The solution was to force the VAE to run in full 32-bit for encoding, then cast the output to 16-bit. In code, I wrapped the VAE encode call in a disabled-autocast block:



```
pythonCopy codewith torch.autocast(device.type, enabled=False, dtype=torch.float32):
    enc = vae_model.encode(pixel_values)
```

This change prevented NaN values by ensuring precise computation during the pixels→latent compression. In short, keeping the VAE in fp32 fixed the instability that 16-bit caused, at a small cost of extra memory which the 16GB could handle.



### 8-bit AdamW vs. Standard AdamW

Instead of the default AdamW optimizer, I used bitsandbytes’ 8-bit AdamW. The advantage is huge memory savings: typically ~4× less memory, which aligns with documentation stating ~75% memory reduction(https://huggingface.co/docs/bitsandbytes/v0.43.0/en/optimizers). On a model with billions of parameters, full 32-bit AdamW would have required enormous RAM for the optimizer’s state (e.g. hundreds of GB for 8B parameters). With 8-bit AdamW, those states are quantized, so I could train effectively on 16 GB. Hugging Face’s bitsandbytes docs explicitly note that 8-bit AdamW preserves performance while slashing memory (75% less footprint). In practice, this enabled the training to run without GPU memory errors.



### Inference Memory Optimizations

For inference (generation), I also needed tricks:



- **VAE Slicing:** Even after training, generating images was tricky. The final VAE *decoding* step (latent→pixel) can blow up memory if done in one go. Diffusers provides a “sliced VAE” mode: calling `pipe.enable_vae_slicing()` causes the VAE to decode the image in smaller patches sequentially (https://huggingface.co/docs/diffusers/v0.26.2/en/optimization/memory). I enabled this, so instead of allocating one giant tensor, the VAE processes the image slice-by-slice. This adds a bit of computation time but prevents the 16 GB GPU from OOM-ing in the last step. In short, VAE slicing traded time for a huge drop in peak memory.
- **CPU Offloading (Attempted):** I experimented with `pipe.enable_sequential_cpu_offload()`, which moves parts of the model to CPU during inference to save GPU RAM. The docs show this can reduce VRAM dramatically (https://huggingface.co/docs/diffusers/v0.26.2/en/optimization/memory#:~:text=To perform CPU offloading%2C call,enable_sequential_cpu_offload), but each diffusion step then swaps submodules on and off the GPU, making inference *very* slow (https://huggingface.co/docs/diffusers/v0.26.2/en/optimization/memory#:~:text=CPU offloading works on submodules,large number of memory transfers). In my tests, CPU offload let the model fit, but generation became prohibitively slow and still sometimes OOM’d (since after one pass it had to keep swapping for the next prompt). Ultimately I chose to keep everything on GPU but load it carefully (see next point).
- **Low-CPU-Memory Loading:** To avoid Colab’s RAM crash when loading all components together, I used `low_cpu_mem_usage=True` on the `from_pretrained` calls. This parameter (documented by Hugging Face) loads models directly onto the GPU with minimal CPU buffering. By loading each submodule (VAE, UNet, CLIP encoders) with `low_cpu_mem_usage=True` and then assembling the pipeline, I prevented the system RAM from overflowing. In effect, I bypassed the normal cache-heavy load to keep the CPU free and rely on GPU memory.

### Storage & Loading Strategy

Another practical issue was saving and loading checkpoints. Re-downloading the full SDXL model each time was slow. To speed this up, I saved model parts individually to Google Drive (e.g. `/unet`, `/vae` folders). Since the default `StableDiffusionXLPipeline.from_pretrained()` expects a `model_index.json`, I wrote a small custom loader that points each component to its Drive location and then constructs the pipeline manually. This let me reuse already-downloaded weights without conforming to a single directory layout.



### Silent LoRA Failure (Name Mismatch)

Initially, after training, the fine-tuned model appeared to produce *the same* images as the base SDXL model – clearly the Naruto style wasn’t being applied. I realized this was due to a naming/namespace mismatch between how PEFT saved the LoRA keys and how Diffusers expected them. The fix was a “hard” weight injection: wrapping the original UNet in a `PeftModel` and then merging the weights. In practice, I did:



```
pythonCopy codepipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_WEIGHTS_PATH)
pipe.unet = pipe.unet.merge_and_unload()
```

This forces-drops the LoRA weights directly into the UNet. Hugging Face’s documentation on merging LoRAs describes this exact process (create a PeftModel and merge it) as the recommended way to fuse LoRA adapters into the base weights (https://huggingface.co/docs/diffusers/main/en/using-diffusers/merge_loras). After this step, the style transfer was correctly applied in the generated images.



### Gradient Checkpointing

Finally, to save memory during backpropagation, I enabled gradient checkpointing on the UNet. This technique does *not* store all intermediate activations. Instead, it re-computes them on the backward pass as needed, which can roughly halve the peak memory usage. According to performance guides, gradient checkpointing can cut VRAM by ~30–50% (with a 15–25% speed penalty) (https://www.runpod.io/articles/guides/scaling-stable-diffusion-training-on-runpod-multi-gpu-infrastructure#:~:text=match at L286 Gradient checkpointing,memory savings without quality loss). 

In practice, enabling `unet.enable_gradient_checkpointing()` was essential to fit the training on 16 GB VRAM. The extra computation was an acceptable trade-off for the ability to run the larger model.



## Conclusion

By combining multiple layers of optimization, I successfully fine-tuned SDXL on a constrained 16 GB GPU. Algorithmic techniques (LoRA and gradient checkpointing) and engineering tricks (8-bit AdamW, VAE slicing, careful memory loading) worked together to keep memory under control. The end result is a fine-tuned model that generates Naruto-style images with high fidelity, all without sacrificing the SDXL base’s generative quality. This project demonstrates that even a 2.6B-parameter model can be trained and used on consumer-grade hardware with careful engineering.
