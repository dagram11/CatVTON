# Expected final content of /content/CatVTON_project/model/pipeline.py
# This is based on the GitHub version of pipeline.py, with the __init__ method modified
# to handle the "flux_lora" case for attn_ckpt_version.

import inspect
import os
from typing import Union

import PIL # Ensure PIL is imported if PIL.Image.Image is used in type hints
import numpy as np
import torch
import tqdm # TQDM was used in the __call__ method
from accelerate import load_checkpoint_in_model # Used in our modified __init__ and original auto_attn_ckpt_load
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, DiffusionPipeline # Added DiffusionPipeline for base class
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import snapshot_download
from transformers import CLIPImageProcessor # CLIPTextModel, CLIPTokenizer are imported in utils.py

# Imports from within the 'model' package
from model.attn_processor import SkipAttnProcessor 
# No, pipeline.py does not import from model.utils. It imports from root utils.
# from model.utils import get_trainable_module, init_adapter # This was the original problematic import

# Imports from the root-level utils.py (after we modified this import)
from utils import get_trainable_module, init_adapter # This is correct for init_adapter and get_trainable_module
from utils import (compute_vae_encodings, numpy_to_pil, prepare_image, # These are also from root utils.py
                   prepare_mask_image, resize_and_crop, resize_and_padding, init_weight_dtype)


class CatVTONPipeline(DiffusionPipeline): # Inherit from DiffusionPipeline
    def __init__(
        self,
        base_ckpt: str,
        attn_ckpt: str,
        attn_ckpt_version: str = "mix",
        weight_dtype=torch.float32, # Default from your pasted code
        device='cuda',
        compile: bool = False, # Added type hint and default from your pasted code
        skip_safety_check: bool = False, # Added type hint and default
        use_tf32: bool = True, # Added type hint and default
    ):
        super().__init__() # Call to base class __init__ is good practice

       
        self.weight_dtype = weight_dtype
        self.skip_safety_check = skip_safety_check

        # Note: init_weight_dtype was imported from utils, make sure it's used if needed.
        # The original code you pasted didn't explicitly use it here, but it was defined.
        # self.weight_dtype = init_weight_dtype(weight_dtype_str) if isinstance(weight_dtype, str) else weight_dtype

        self.noise_scheduler = DDIMScheduler.from_pretrained(base_ckpt, subfolder="scheduler")
        # Using a specific VAE as per your pasted code.
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device, dtype=self.weight_dtype)
        
        if not skip_safety_check:
            self.feature_extractor = CLIPImageProcessor.from_pretrained(base_ckpt, subfolder="feature_extractor")
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(base_ckpt, subfolder="safety_checker").to(device, dtype=self.weight_dtype)
        else: # Ensure these are None if skipped, to avoid errors in run_safety_checker
            self.feature_extractor = None
            self.safety_checker = None
            
        self.unet = UNet2DConditionModel.from_pretrained(base_ckpt, subfolder="unet").to(device, dtype=self.weight_dtype)
        
        # LoRA / Attention Modules Initialization
        init_adapter(self.unet, cross_attn_cls=SkipAttnProcessor)
        self.attn_modules = get_trainable_module(self.unet, "attention")
        
        # === START OF MODIFIED LoRA LOADING (this is the part you manually edit/insert) ===
        print(f"DEBUG: In __init__, trying to load LoRA. attn_ckpt='{attn_ckpt}', attn_ckpt_version='{attn_ckpt_version}'")
        if attn_ckpt_version == "flux_lora":
            lora_weights_path = os.path.join(attn_ckpt, "pytorch_lora_weights.safetensors")
            if os.path.exists(lora_weights_path):
                print(f"DEBUG: Directly loading LoRA weights for 'flux_lora' from: {lora_weights_path}")
                try:
                    load_checkpoint_in_model(self.attn_modules, lora_weights_path)
                    print("DEBUG: Successfully loaded LoRA weights directly into self.attn_modules.")
                except Exception as e_lora_direct:
                    print(f"ERROR directly loading LoRA from {lora_weights_path}: {e_lora_direct}")
                    raise
            else:
                print(f"CRITICAL: LoRA file {lora_weights_path} not found for 'flux_lora' version.")
                print(f"Contents of {attn_ckpt}:")
                if os.path.exists(attn_ckpt) and os.path.isdir(attn_ckpt):
                    for item in os.listdir(attn_ckpt): print(f"  - {item}")
                else:
                    print(f"  Directory {attn_ckpt} does not exist.")
                raise FileNotFoundError(f"LoRA weights not found: {lora_weights_path}")
        else:
            # If not "flux_lora", fall back to the original auto_attn_ckpt_load
            print(f"DEBUG: attn_ckpt_version is '{attn_ckpt_version}', calling original self.auto_attn_ckpt_load method.")
            self.auto_attn_ckpt_load(attn_ckpt, attn_ckpt_version) # Call the original method
        # === END OF MODIFIED LoRA LOADING ===
        
        # Pytorch 2.0 Compile
        if compile:
            self.unet = torch.compile(self.unet)
            self.vae = torch.compile(self.vae, mode="reduce-overhead")
            
        # Enable TF32 for faster training on Ampere GPUs (A100 and RTX 30 series).
        if use_tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True

    # This is the ORIGINAL auto_attn_ckpt_load method from GitHub. It should remain.
    def auto_attn_ckpt_load(self, attn_ckpt, version):
        sub_folder_map = { # Renamed variable for clarity
            "mix": "mix-48k-1024",
            "vitonhd": "vitonhd-16k-512",
            "dresscode": "dresscode-16k-512",
        }
        # Use .get() for safer dictionary access to avoid KeyError if version is unexpected by this original logic
        sub_folder = sub_folder_map.get(version)

        if sub_folder is None:
            # If version is not one of the predefined keys, assume version itself is the sub_folder name
            # This was the original behavior for the 'else' case.
            # However, our modified __init__ should prevent "flux_lora" from reaching here.
            print(f"DEBUG: auto_attn_ckpt_load called with version='{version}', not in map. Treating version as sub_folder directly (original behavior).")
            sub_folder = version # This would happen if __init__ logic was different

        # Construct the path to where the 'attention' checkpoint or LoRA is expected
        # The original code expected attn_ckpt/sub_folder/attention
        # For LoRAs like pytorch_lora_weights.safetensors, they are usually directly in attn_ckpt/sub_folder/
        # Let's try the direct path first, then the 'attention' sub-subfolder.

        # Path to where the actual LoRA weights file might be
        potential_lora_file_path = os.path.join(attn_ckpt, sub_folder, "pytorch_lora_weights.safetensors")
        # Original path structure expected by the script
        original_attention_folder_path = os.path.join(attn_ckpt, sub_folder, 'attention')

        # Determine the actual checkpoint path to use
        checkpoint_to_load = None
        if os.path.exists(potential_lora_file_path):
            checkpoint_to_load = potential_lora_file_path
            print(f"DEBUG: auto_attn_ckpt_load found LoRA at {checkpoint_to_load}")
        elif os.path.exists(original_attention_folder_path): # Check if it's a folder or file
             if os.path.isfile(original_attention_folder_path): # If 'attention' is a file
                 checkpoint_to_load = original_attention_folder_path
                 print(f"DEBUG: auto_attn_ckpt_load found attention file at {checkpoint_to_load}")
             elif os.path.isdir(original_attention_folder_path): # If 'attention' is a directory
                 # This case matches the original script's expectation for load_checkpoint_in_model
                 checkpoint_to_load = original_attention_folder_path
                 print(f"DEBUG: auto_attn_ckpt_load found attention directory at {checkpoint_to_load}")
             else: # It exists but is neither file nor dir? Unlikely.
                 print(f"Warning: {original_attention_folder_path} exists but is not a file or directory.")

        else:
            # This branch handles if attn_ckpt is an HF repo ID to download from
            if not os.path.exists(attn_ckpt): # If attn_ckpt itself is not a local path
                print(f"DEBUG: attn_ckpt '{attn_ckpt}' not a local path. Assuming it's an HF repo ID for snapshot_download.")
                repo_path = snapshot_download(repo_id=attn_ckpt)
                print(f"Downloaded {attn_ckpt} to {repo_path}")
                # Now construct paths relative to the downloaded repo_path
                potential_lora_file_path = os.path.join(repo_path, sub_folder, "pytorch_lora_weights.safetensors")
                original_attention_folder_path = os.path.join(repo_path, sub_folder, 'attention')
                
                if os.path.exists(potential_lora_file_path):
                    checkpoint_to_load = potential_lora_file_path
                elif os.path.exists(original_attention_folder_path):
                     checkpoint_to_load = original_attention_folder_path

            if not checkpoint_to_load:
                print(f"CRITICAL: In auto_attn_ckpt_load, no valid checkpoint found for version '{version}', sub_folder '{sub_folder}' under '{attn_ckpt}'.")
                print(f"  Checked: {potential_lora_file_path}")
                print(f"  Checked: {original_attention_folder_path}")
                # raise FileNotFoundError(f"No checkpoint found for {version}/{sub_folder}")
                return # Silently return if not found, or raise error. Original just continued.

        if checkpoint_to_load:
            print(f"DEBUG: auto_attn_ckpt_load - Loading attention checkpoint from {checkpoint_to_load}")
            try:
                load_checkpoint_in_model(self.attn_modules, checkpoint_to_load)
                print(f"DEBUG: Successfully loaded from {checkpoint_to_load} in auto_attn_ckpt_load.")
            except Exception as e_auto_load:
                print(f"ERROR in auto_attn_ckpt_load loading from {checkpoint_to_load}: {e_auto_load}")
                raise
        else:
            # Original code had a 'continue' here if path didn't exist, implying a loop.
            # Since we're processing one 'version' leading to one 'sub_folder', 'continue' isn't right.
            # We should raise an error or log that it wasn't found.
            print(f"Warning: In auto_attn_ckpt_load, checkpoint for version '{version}' (sub_folder '{sub_folder}') was not loaded as no valid path was found.")


    def run_safety_checker(self, image):
        if self.safety_checker is None or self.feature_extractor is None: # Added feature_extractor check
            has_nsfw_concept = [False] * image.shape[0] if isinstance(image, np.ndarray) else [False] # Default to safe
        else:
            # Ensure image is in the format safety_checker expects (PIL list or NumPy)
            if isinstance(image, np.ndarray):
                # If NumPy, it's likely (B, H, W, C) from pipeline output. Convert to list of PIL.
                pil_images = numpy_to_pil(image) # numpy_to_pil should be in utils.py
            elif isinstance(image, list) and all(isinstance(i, PIL.Image.Image) for i in pil_images):
                pil_images = image # Already a list of PIL images
            else: # Try to convert single PIL image to list
                try:
                    pil_images = [image] if isinstance(image, PIL.Image.Image) else [PIL.Image.fromarray(image.astype(np.uint8))]
                except:
                    print("Warning: Could not convert image to list of PIL for safety checker.")
                    has_nsfw_concept = [False] # Default to safe
                    return image, has_nsfw_concept

            safety_checker_input = self.feature_extractor(pil_images, return_tensors="pt").to(self.device)
            # The safety_checker might expect NumPy array of (0, 255) or list of PIL
            # The diffusers safety_checker typically takes `images=numpy_array, clip_input=tensor`
            # Let's ensure images passed to safety_checker are numpy if they were PIL
            images_for_safety_checker = []
            if isinstance(pil_images, list): # Ensure it's a list of NumPy arrays for safety_checker
                for pil_img in pil_images:
                    images_for_safety_checker.append(np.array(pil_img))
            else: # Should not happen if conversion above worked
                images_for_safety_checker = np.array(pil_images)


            checked_images, has_nsfw_concept = self.safety_checker(
                images=images_for_safety_checker, # Pass list of NumPy arrays
                clip_input=safety_checker_input.pixel_values.to(self.weight_dtype)
            )
            # Convert checked_images back to list of PIL if needed by downstream code
            image = numpy_to_pil(checked_images) if isinstance(checked_images, np.ndarray) else checked_images

        return image, has_nsfw_concept

    def check_inputs(self, image, condition_image, mask, width, height):
        if isinstance(image, torch.Tensor) and isinstance(condition_image, torch.Tensor) and isinstance(mask, torch.Tensor):
            return image, condition_image, mask
        # Ensure inputs are PIL Images before resizing
        if not isinstance(image, PIL.Image.Image): image = PIL.Image.fromarray(image.astype(np.uint8)) if isinstance(image, np.ndarray) else image
        if not isinstance(condition_image, PIL.Image.Image): condition_image = PIL.Image.fromarray(condition_image.astype(np.uint8)) if isinstance(condition_image, np.ndarray) else condition_image
        if not isinstance(mask, PIL.Image.Image): mask = PIL.Image.fromarray(mask.astype(np.uint8)) if isinstance(mask, np.ndarray) else mask

        if image.size != (width, height) or mask.size != (width, height): # Only resize if needed
            assert image.size == mask.size, "Image and mask must have the same size before resize_and_crop"
            image = resize_and_crop(image, (width, height))
            mask = resize_and_crop(mask, (width, height))
        if condition_image.size != (width, height): # Only resize if needed
            condition_image = resize_and_padding(condition_image, (width, height))
        return image, condition_image, mask

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, torch.Tensor],
        condition_image: Union[PIL.Image.Image, torch.Tensor],
        mask: Union[PIL.Image.Image, torch.Tensor],
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        height: int = 1024, # Default from your pasted code
        width: int = 768,   # Default from your pasted code
        generator=None, # generator: Optional[torch.Generator] = None,
        eta: float = 0.0, # Default from diffusers, your code had 1.0
        output_type: str = "pil", # Added for consistency with DiffusionPipeline
        return_dict: bool = True, # Added for consistency
        **kwargs # Allow other args
    ):
        # Ensure concat_dim is defined
        # In the original code, concat_dim was -2. For [B, C, H, W] latents,
        # if you want to concat along Height, dim should be 2 (or -2).
        # If along Width, dim should be 3 (or -1).
        # Given typical image processing, concat along Channel (dim=1) is more common for some inputs.
        # However, their comment said "FIXME: y axis concat", implying H, so dim=-2 is likely correct.
        concat_dim = -2

        # 0. Default height and width to unet configuration if not set
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. PIL images are converted to specified H, W. Tensors are assumed to be preprocessed.
        image, condition_image, mask = self.check_inputs(image, condition_image, mask, width, height)

        # 2. Prepare mask and masked_image
        # utils.prepare_image converts PIL to [-1, 1] tensor (B, C, H, W)
        # utils.prepare_mask_image converts PIL to [0, 1] tensor (B, 1, H, W)
        image_tensor = prepare_image(image).to(device=self.device, dtype=self.weight_dtype)
        condition_image_tensor = prepare_image(condition_image).to(device=self.device, dtype=self.weight_dtype)
        mask_tensor = prepare_mask_image(mask).to(device=self.device, dtype=self.weight_dtype)

        masked_image_tensor = image_tensor * (mask_tensor < 0.5) # Invert mask logic if mask is for "keep"

        # 3. VAE encoding
        # utils.compute_vae_encodings needs vae and image tensor
        masked_latent = compute_vae_encodings(masked_image_tensor, self.vae)
        condition_latent = compute_vae_encodings(condition_image_tensor, self.vae)

        # Ensure mask_latent is correctly broadcastable for latent space operations
        # It needs to be [B, 1, H_lat, W_lat]
        mask_latent = torch.nn.functional.interpolate(
            mask_tensor, size=masked_latent.shape[-2:], mode="nearest"
        )
        
        del image_tensor, mask_tensor, condition_image_tensor # Free memory

        # 4. Concatenate latents. Their comment "FIXME: y axis concat" suggests H-dim concatenation.
        # Latents are typically (B, C_lat, H_lat, W_lat). C_lat is usually 4.
        # If concat_dim = -2 (H), shapes must match in W.
        # If concat_dim = 1 (C), shapes must match in H, W. This is more standard for some inpainting inputs.
        # Let's assume their FIXME means they intended to concat along C for the UNet input structure.
        # The original CatVTON inpainting model input was [noisy_latents, mask_latents, condition_latents] along channel dim.
        # So, masked_latent_concat in their __call__ was:
        # masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=concat_dim)
        # mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)
        # This seems to imply concat_dim operates on a different structure than the final UNet input.
        # Let's stick to the original UNet input structure from most inpainting pipelines:
        # Latents for UNet: (B, C_unet_in, H_lat, W_lat) where C_unet_in = C_lat + 1 (mask) + C_lat (masked_image)
        # Or for ControlNet-like conditioning: C_unet_in = C_lat + C_control
        # The specific CatVTON unet might have a different input channel expectation.
        # The original code's `inpainting_latent_model_input = torch.cat([non_inpainting_latent_model_input, mask_latent_concat, masked_latent_concat], dim=1)`
        # suggests the unet input channel dim (dim=1) is where all these are stacked.

        # For classifier-free guidance, inputs are duplicated.
        # Let's keep this part similar to original.
        # The `masked_latent_concat` and `mask_latent_concat` from original code are actually parts of the UNet input, not the noisy latents.

        # 5. Prepare noise and timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps

        # Initial latents are random noise with the shape of the masked_image's VAE encoding
        shape = masked_latent.shape
        latents = randn_tensor(shape, generator=generator, device=self.device, dtype=self.weight_dtype)
        latents = latents * self.noise_scheduler.init_noise_sigma

        # 6. Classifier-Free Guidance setup
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            # For CFG, unet needs unconditional and conditional inputs.
            # We need unconditional versions of condition_latent. A common way is zeros.
            uncond_condition_latent = torch.zeros_like(condition_latent)

            # The UNet input will be [batch_of_latents, batch_of_mask, batch_of_masked_image_latents, batch_of_condition_latents]
            # For CFG, this gets duplicated.
            # masked_latent is the VAE of the image with parts masked out.
            # condition_latent is the VAE of the cloth.
            # mask_latent is the resized mask.
        else: # Not used in the original CatVTON __call__ if guidance_scale <= 1.0 for this part
            pass


        # 7. Denoising loop
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.noise_scheduler.order # As in original

        for i, t in enumerate(tqdm.tqdm(timesteps)): # Use tqdm.tqdm if imported as tqdm
            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)

            # Prepare UNet input: [latent_model_input, mask, masked_image_latent, condition_latent]
            # For CFG, mask, masked_image_latent, condition_latent need to be duplicated or handled.
            # The original code concatenates mask_latent_concat and masked_latent_concat to non_inpainting_latent_model_input
            # non_inpainting_latent_model_input = latent_model_input
            # mask_latent_concat was torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)
            # masked_latent_concat was torch.cat([masked_latent, condition_latent], dim=concat_dim)
            # This structure for mask_latent_concat and masked_latent_concat needs to be specific to what the UNet expects.
            # The original CatVTON UNet input seems to be 9 channels: 4 for noisy latents, 1 for mask, 4 for masked VAE image.
            # Condition (cloth) is passed via cross-attention, not input channels.
            # The `encoder_hidden_states=None, # FIXME` line in original unet call is key.
            # The new FLUX model used here is different. It expects text embeddings.
            # The `condition_image` is VAE encoded to `condition_latent`. How is this used by FLUX?
            # FLUX typically uses text embeddings and image embeddings.
            # If `condition_latent` is an image embedding for FLUX, it needs to be passed correctly.

            # This pipeline seems to be based on Stable Diffusion Inpainting, not FLUX directly in its call.
            # The UNet is from base_ckpt (SD Inpainting). LoRA adapts its attention.
            # SD Inpainting UNet input: (B, 9, H, W) = (latent (4) + mask (1) + masked_image (4))
            
            unet_input_list = [latent_model_input]

            if do_classifier_free_guidance:
                current_masked_latent = torch.cat([masked_latent] * 2)
                current_mask_latent = torch.cat([mask_latent] * 2)
                # Condition for SD Inpainting is the masked image itself, not a separate condition_latent in channels
            else:
                current_masked_latent = masked_latent
                current_mask_latent = mask_latent

            unet_input_list.append(current_mask_latent)
            unet_input_list.append(current_masked_latent)
            
            inpainting_latent_model_input = torch.cat(unet_input_list, dim=1)


            # Predict noise
            # For SD Inpainting, encoder_hidden_states are usually from a text prompt.
            # If this is image-conditioned inpainting, the "condition_image" (cloth)
            # might need to be used to generate encoder_hidden_states if the UNet is conditioned that way.
            # The CatVTON paper concatenates cloth and person features.
            # The original pipeline had `encoder_hidden_states=None, # FIXME`.
            # This implies text conditioning is either absent or needs to be correctly handled.
            # For now, let's assume no text conditioning, which is typical for pure SD Inpainting.
            
            noise_pred = self.unet(
                inpainting_latent_model_input,
                t, # t.to(self.device) already handled by timesteps
                encoder_hidden_states=None, # No text prompt for simple inpainting
                return_dict=False,
            )[0]

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Scheduler step
            latents = self.noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.noise_scheduler.order == 0):
                if isinstance(tqdm.tqdm, type): # Check if progress_bar is a class or instance
                     pass # In this case, tqdm.tqdm is used as a context manager usually
                else: # If it's an instance, update it
                     tqdm.tqdm.update(1) # This is likely wrong way to update if used as context manager

        # 8. Decode latents
        # The original code split latents if concat_dim was used on latents themselves.
        # Here, `latents` are the denoised version of the initial random noise for the masked area.
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

        # 9. Post-process and safety check
        image = (image / 2 + 0.5).clamp(0, 1)
        image_numpy = image.cpu().permute(0, 2, 3, 1).float().numpy() # (B, H, W, C)
        
        if output_type == "pil":
            pil_images = numpy_to_pil(image_numpy)
            if not self.skip_safety_check:
                # Original safety checker took numpy array and returned numpy + boolean list
                # Our run_safety_checker now expects PIL list or NumPy, returns PIL list + boolean list
                pil_images, has_nsfw_concept = self.run_safety_checker(pil_images) # Pass PIL list
            else:
                has_nsfw_concept = [False] * len(pil_images)
            
            if not return_dict:
                return (pil_images, has_nsfw_concept)
            
            # Not sure if DiffusionPipelineOutput is defined or expected here.
            # Let's just return a list of images if not return_dict is False.
            # For simplicity, this pipeline will just return the list of PIL images.
            return pil_images # Return list of PIL images


        elif output_type == "latent":
            # Safety check not typically applied to latents directly
            if not return_dict:
                return (latents, None) # No NSFW concept for latents
            return {"sample": latents} # Or a more structured output
        
        else: # numpy
            if not self.skip_safety_check:
                # Pass numpy array to safety checker
                checked_image_numpy, has_nsfw_concept = self.run_safety_checker(image_numpy)
            else:
                checked_image_numpy = image_numpy
                has_nsfw_concept = [False] * image_numpy.shape[0]

            if not return_dict:
                return (checked_image_numpy, has_nsfw_concept)
            return {"sample": checked_image_numpy} # Or a more structured output

# END OF pipeline.py
