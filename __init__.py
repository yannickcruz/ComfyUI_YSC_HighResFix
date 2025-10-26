import torch
import numpy as np
import comfy.model_management as model_management
import comfy.samplers
import comfy.utils
import comfy.sd
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

class YSC_HighResFix:

    def __init__(self):
        self.device = model_management.get_torch_device()
        self.image_scaler = ImageUpscaleWithModel()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "positive": ("CONDITIONING"),
                "negative": ("CONDITIONING"),
                "upscale_factor": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 4.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "upscale_method": (["nearest-exact", "bilinear", "bicubic", "lanczos"],),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 1000
                }),
                "cfg": ("FLOAT", {
                    "default": 8.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 0.1,
                    "round": 0.01
                }),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES,),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,),
                "denoise": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 1.01
                }),
                "upscale_model": ("UPSCALE_MODEL",),
            },
            "optional": []
        }
    
    RETURN_TYPES = ("LATENT")
    RETURN_NAMES = ("LATENT")
    FUNCTION = "refine_upscale"
    CATEGORY = "image/upscaling"


    def image_upscale(self, upscale_model, image, upscale_factor, upscale_method):
        samples = image.movedim(-1,1)

        width = round(samples.shape[3] * upscale_factor)
        height = round(samples.shape[2] * upscale_factor)

        samples = self.__imageScaler.upscale(upscale_model, image)[0].movedim(-1,1)

        upscaled_width = round(samples.shape[3])
        upscaled_height = round(samples.shape[2])

        if upscaled_width > width or upscaled_height > height:
            samples = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
            
        samples = samples.movedim(1,-1)
        return (samples,)
    
# Registro do nó
NODE_CLASS_MAPPINGS = {
    "YSC_HighresFix": YSC_HighresFix
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YSC_HighresFix": "Highres Fix (YSC)"
}
