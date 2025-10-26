# ComfyUI YSC HighRes-Fix

A custom node for ComfyUI designed to simplify the "High-Resolution Fix" process. This node takes an input image, performs a direct upscale, and then re-encodes it into a latent image, preparing it for a final sampling pass to add details.

## What It Does

This node replicates the functionality of a high-resolution fix pipeline. Instead of just resizing an image, it intelligently upscales it and then uses a diffusion sampling process (denoising) to add fine details that are consistent with the original image's content.

## How It Works

The node's operation is straightforward:

1.  **Image Input:** You provide a low-resolution image (e.g., the output of your initial generation).
2.  **Brute Upscale:** The node applies a simple, direct upscale to the image to increase its dimensions.
3.  **Latent Encoding:** This larger, upscaled image is then encoded back into the latent space.
4.  **Latent Output:** The node outputs this new, larger latent image.

This output latent is then ready to be passed to a `KSampler` node (or similar) with a low-to-moderate denoising strength. This final sampling step is where the new details are "painted" onto the upscaled image.

## How to Use (Example Workflow)

1.  Generate your initial image (e.g., 512x512) as you normally would.
2.  Connect the output image from your `VAE Decode` node to the `image` input of the **YSC HighRes-Fix** node.
3.  Connect the `LATENT` output from the **YSC HighRes-Fix** node to the `latent_image` input of a second `KSampler` node.
4.  Connect your model, positive prompt, and negative prompt to this second `KSampler`.
5.  **Crucially, set the `denoise` on this second `KSampler` to a value like `0.3` to `0.55`.** This value controls how much detail is added.
6.  Connect the `LATENT` output of this `KSampler` to a final `VAE Decode` node to get your high-resolution image.

## Inputs

* `image`: The source image (from a `VAE Decode` or `Load Image` node) that you want to upscale and refine.

## Outputs

* `LATENT`: The upscaled and re-encoded latent image, ready for a final sampling (denoising) pass.

## Installation

1.  Navigate to your `ComfyUI/custom_nodes/` directory.
2.  Clone this repository:
    ```bash
    git clone https://github.com/yannickcruz/ComfyUI-YSC-HighResFix.git
    ```
    (Replace the URL with your actual repository URL)
3.  Alternatively, download the `.py` files from this repository and place them directly in the `ComfyUI/custom_nodes/` directory.
4.  Restart ComfyUI.
