# Rasengan-XL - Technical Report

Fine Tune Model - GDrive - https://drive.google.com/drive/folders/1Ns17jEBQJu7kL5i7-9HUvAvgQRJjpmJj?usp=sharing


# Rasengan-XL

This technical report documents the engineering process, architectural decisions, and optimization strategies employed to fine-tune the Stable Diffusion XL (SDXL) 1.0 model on  NVIDIA Tesla T4 GPU (16GB VRAM) using Google Colab .The objective was to adapt the 2.6-billion parameter base model to generate high quality images in the specific art style of the Naruto anime series, utilizing the lambdalabs/naruto-blip-captions dataset.

This report details the end-to-end pipeline from training to inference, with a specific focus on overcoming significant hardware constraints through various techniques.

**Technical Report**: Memory-Efficient Fine-Tuning of SDXL on T4 GPU.

Training Methodology

So for the efficient training I used Parameter-Efficient Fine-Tuning (PEFT) using Low-Rank Adaptation (LoRA). Instead of retraining the full parameters of the UNet which would requie more than 20 GB of VRAM, I injected small, trainable rank-decomposition matrices into specific attention layers of the model while freezing the pre-trained weights.

## Model Architecture & ConfigurationBase Model:

Stability AI SDXL Base 1.0.Dataset: lambdalabs/naruto-blip-captions (1,221 image-text pairs).
Resolution: 512x512 pixels (Optimized for T4 VRAM limits).
Precision: Mixed Precision (fp16 for UNet/Text Encoders, fp32 for VAE).

LoRA ConfigurationWe utilized the PEFT library to inject adapters.
Rank ($r$): 16. This controls the dimension of the trainable matrices.A rank of 16 provides a balance between parameter efficiency and stylistic fidelity.
Alpha ($\alpha$): 32. The scaling factor for the learned weights.
Target Modules: ["to_q", "to_k", "to_v", "to_out.0"]. These values were used as they specifically target the Self-Attention and Cross-Attention projection layers of the UNet. These layers are responsible for mapping the relationships between visual features and text prompts, making them the most effective targets for style transfer.
Trainable Parameters: This configuration resulted in only ~23 million trainable parameters out of the total 2.6 billion (approx. 0.9%), significantly reducing VRAM requirements for gradients and optimizer states.
Training Loop Implementation:
The training loop was built to handle specific memory constraints:
Data Loading: Made sure that the image were properly pre-processed (resized, center-cropped, normalized) and tokenized on the CPU.
VAE Encoding: Images were encoded into latents using the VAE in float32 to ensure numerical stability, then cast to float16 for the diffusion process.
Diffusion Process: Noise was added to latents, and the LoRA-augmented UNet predicted the noise residual.
Optimization: Gradients were calculated only for the LoRA adapters and updated using the 8-bit AdamW optimizer.

## Engineering Design Choices & Justifications
In this I have tried to explain why I used what in the entire training process.
The VAE Precision Dilemma: Why Float32?

The problem I faced during training initialization, was that the SDXL VAE (Variational Autoencoder) was showing  NaN (Not a Number).These values were propogating throught the entire network and caused the training to fail. What I was doing was that I was loading the model on GPU and using float16 as the data type. It turns out that it is usntable on this data type and many users on github had also reported the same issue. So in order to solve this I enforced a strict datatype policy. While the massive UNet was loaded in fp16 to save memory, the VAE was loaded and executed in float32 (Full Precision) on the GPU. This was implemented in the training loop where I disabled the automatic mixed precision for the VAE encoding step.
```python
with torch.autocast(device.type, enabled=False, dtype=torch.float32):
    enc = vae_model.encode(pixel_values)
    
```

This ensured that the compression from pixels to latents was mathematically precise, preventing numerical overflow.

Optimization Strategy:

8-bit AdamWThe Standard:
Why the 8bit AdamW over the standard one?
Well using 8-bit AdamW over standard AdamW primarily offers significant memory savings and can improve training speed, making it highly beneficial for training or fine-tuning large models on memory-constrained hardware.
Standard AdamW requires storing optimizer states (first and second moments) in full precision (typically float32), consuming approximately 8 bytes per model parameter, which can double the memory requirement compared to the model itself.
For a model with 8 billion parameters, this translates to roughly 64 GB of GPU memory dedicated solely to optimizer states.
Basically this doc helped a lot : https://huggingface.co/docs/bitsandbytes/v0.43.0/en/optimizers

So using the above technique quantizes the optimizer states into 8-bit integers, reducing the memory footprint of the optimizer by approximately 75%.

VAE Slicing During Inference:
So during the inference I was wasting a lot of time as I was using the CPU for it. Also when I used GPU, it caused the OOM error. So I came across this : https://github.com/vladmandic/sdnext/issues/3416

Was kinda facing the same problem, these people just put that into words for me. So used this source : https://huggingface.co/docs/diffusers/v0.26.2/en/optimization/memory to understand the issue.
So basically what was happening was that during inference the final step involves the VAE decoding the latent representation  back into a full pixel image . This convolution operation requires a massive temporary tensor allocation that typically exceeds the 16GB VRAM buffer of a T4 GPU, causing an OutOfMemoryError at the very last second.

So to solve this I implemented VAE Slicing:

```python
  (pipe.enable_vae_slicing())

```
Instead of decoding the entire latent tensor at once, this technique splits the image into smaller "slices," decodes them sequentially, and stitches the output back together. This trades a small amount of computation time for a massive reduction in peak VRAM usage, preventing the crash.

Also as mentioned above intially I  attempted to use Sequential CPU Offloading (enable_sequential_cpu_offload) to do inferencingon the CPU directly.Though this allowed the model to fit by keeping weights in System RAM (CPU), it was slow and after generating a pair of image - one from the normal and one from the fine tuned model it used to run out of memory. So i decided to simply use the GPU for inferencing. In that as well I managed the memory manually by using 

```python
  low_cpu_mem_usage=True
```
during the the loading process. This reduced inference time by a lottttt!

# Some issue and their solutions:
So loading all the components all together lead to system RAM hitting its limit and the colab runtime crashing as well. In order to handle this I utilized 
```python
  low_cpu_mem_usage=True
```
(can be found at : https://huggingface.co/docs/diffusers/v0.26.2/en/optimization/) during the from_pretrained calls.

Storage Optimizations - cause downlaoding the model again and again when runtime used to stop was a pain: To optimize storage, I saved model components (/unet, /vae) individually to Google Drive. However, the standard StableDiffusionXLPipeline.from_pretrained() method expects a model_index.json file to map these components, which was missing.So I wrote a basic pipeline which instantiated each class (CLIPTextModel, UNet2DConditionModel, etc.) individually by pointing them to their specific subfolders on Google Drive, and then manually injected them into the Pipeline constructor. This decoupled my storage structure from the library's rigid directory requirements.

Silent LoRA Failure (Generic Image Output)Issue:
So another thing that I observed during inference was that the basic model and the fine tuned model were generating the same outputs initially. Shocked was I.
After GPTing enough I understood that there was a namespace  mismatch between the peft training keys and the diffusers inference keys.To solve this I used a Hard Weight Injection strategy. Instead of relying on the pipeline to map weights, I used the PeftModel library to wrap the UNet directly



```python
  pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_WEIGHTS_PATH)
  pipe.unet = pipe.unet.merge_and_unload()

```
This mathematically fused the LoRA weights into the base model weights, guaranteeing that the style was applied.

Optimizing the gradient memory: 

Training a 2.6B parameter model requires storing activations for backpropagation, which scales linearly with model depth.So I used Gradient Checkpointing 
```python
  (unet.enable_gradient_checkpointing())
```
This technique does not store all intermediate activations during the forward pass; instead, it re-computes them on the fly during the backward pass which also reduced the VRAM usage.

Conclusion:
Through a combination of algorithmic optimization (LoRA, Gradient Checkpointing) and low-level engineering (Memory Offloading, VAE Slicing, Precision Management), we successfully fine-tuned and deployed a state-of-the-art generative model on constrained hardware. The final model demonstrates a high-fidelity transfer of the Naruto art style while maintaining the generalization capabilities of the base SDXL architecture.