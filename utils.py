# START OF MERGED utils.py
import os
import json # From official
import torch
from model.attn_processor import AttnProcessor2_0, SkipAttnProcessor # From official

# Added from your original utils.py because pipeline.py needs it
# and it was missing from the official Zheng-Chong/CatVTON/main/utils.py
# Ensure PIL.Image is imported if type hints use it (official utils.py imports it)
from PIL import Image # Official utils.py has this, ensure it's here.
                     # Also, many functions below use it.
import PIL # The utils_original.py had this. PIL.Image is usually sufficient.
import math # From official
import numpy as np # From official
from accelerate.state import AcceleratorState # From official
from packaging import version # From official
import accelerate # From official
from typing import List, Optional, Tuple, Set # From official
from diffusers import UNet2DConditionModel, SchedulerMixin # From official (though UNet part might be different in pipeline)
from tqdm import tqdm # From official


# --- Function from your original utils.py (and needed by pipeline.py) ---
def compute_vae_encodings(image: torch.Tensor, vae: torch.nn.Module) -> torch.Tensor:
    """
    Args:
        images (torch.Tensor): image to be encoded
        vae (torch.nn.Module): vae model

    Returns:
        torch.Tensor: latent encoding of the image
    """
    pixel_values = image.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)
    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor
    return model_input

# --- ALL Functions from official Zheng-Chong/CatVTON/main/utils.py START HERE ---
def init_adapter(unet,
                 cross_attn_cls=SkipAttnProcessor,
                 self_attn_cls=None,
                 cross_attn_dim=None,
                 **kwargs):
    if cross_attn_dim is None:
        cross_attn_dim = unet.config.cross_attention_dim
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else cross_attn_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            if self_attn_cls is not None:
                attn_procs[name] = self_attn_cls(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, **kwargs)
            else:
                # retain the original attn processor
                attn_procs[name] = AttnProcessor2_0(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, **kwargs)
        else:
            attn_procs[name] = cross_attn_cls(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, **kwargs)

    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    return adapter_modules

def init_diffusion_model(diffusion_model_name_or_path, unet_class=None):
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer

    text_encoder = CLIPTextModel.from_pretrained(diffusion_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(diffusion_model_name_or_path, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(diffusion_model_name_or_path, subfolder="tokenizer")
    try:
        unet_folder = os.path.join(diffusion_model_name_or_path, "unet")
        unet_configs = json.load(open(os.path.join(unet_folder, "config.json"), "r"))
        unet = unet_class(**unet_configs)
        unet.load_state_dict(torch.load(os.path.join(unet_folder, "diffusion_pytorch_model.bin"), map_location="cpu"), strict=True)
    except:
        unet = None
    return text_encoder, vae, tokenizer, unet

def attn_of_unet(unet):
    attn_blocks = torch.nn.ModuleList()
    for name, param in unet.named_modules():
        if "attn1" in name:
            attn_blocks.append(param)
    return attn_blocks

def get_trainable_module(unet, trainable_module_name):
    if trainable_module_name == "unet":
        return unet
    elif trainable_module_name == "transformer":
        trainable_modules = torch.nn.ModuleList()
        for blocks in [unet.down_blocks, unet.mid_block, unet.up_blocks]:
            if hasattr(blocks, "attentions"):
                trainable_modules.append(blocks.attentions)
            else:
                for block in blocks:
                    if hasattr(block, "attentions"):
                        trainable_modules.append(block.attentions)
        return trainable_modules
    elif trainable_module_name == "attention":
        attn_blocks = torch.nn.ModuleList()
        for name, param in unet.named_modules():
            if "attn1" in name:
                attn_blocks.append(param)
        return attn_blocks
    else:
        raise ValueError(f"Unknown trainable_module_name: {trainable_module_name}")

def init_weight_dtype(wight_dtype): # Note: official repo has typo "wight_dtype"
    return {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[wight_dtype]


def prepare_eval_data(dataset_root, dataset_name, is_pair=True): # From official
    # ... (content of prepare_eval_data from official utils.py) ...
    # This function is long, I'll omit its body for brevity here,
    # but it MUST be included in your actual merged file.
    # For this specific error, it's not the cause, but good to have the full file.
    # For now, let's just ensure the functions pipeline.py *directly* imports are present.
    pass # Placeholder if you don't paste the full body for now

def repaint_result(result, person_image, mask_image): # From official
    result, person, mask = np.array(result), np.array(person_image), np.array(mask_image)
    mask = np.expand_dims(mask, axis=2)
    mask = mask / 255.0
    result_ = result * mask + person * (1 - mask)
    return Image.fromarray(result_.astype(np.uint8))


def prepare_image(image): # From official AND your original utils.py
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(dtype=torch.float32)
    else:
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    return image


def prepare_mask_image(mask_image): # From official AND your original utils.py
    if isinstance(mask_image, torch.Tensor):
        if mask_image.ndim == 2:
            mask_image = mask_image.unsqueeze(0).unsqueeze(0)
        elif mask_image.ndim == 3 and mask_image.shape[0] == 1:
            mask_image = mask_image.unsqueeze(0)
        elif mask_image.ndim == 3 and mask_image.shape[0] != 1:
            mask_image = mask_image.unsqueeze(1)
        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
    else:
        if isinstance(mask_image, (PIL.Image.Image, np.ndarray)):
            mask_image = [mask_image]
        if isinstance(mask_image, list) and isinstance(mask_image[0], PIL.Image.Image):
            mask_image = np.concatenate(
                [np.array(m.convert("L"))[None, None, :] for m in mask_image], axis=0
            )
            mask_image = mask_image.astype(np.float32) / 255.0
        elif isinstance(mask_image, list) and isinstance(mask_image[0], np.ndarray):
            mask_image = np.concatenate([m[None, None, :] for m in mask_image], axis=0)
        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
        mask_image = torch.from_numpy(mask_image)
    return mask_image


def numpy_to_pil(images): # From official AND your original utils.py
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def tensor_to_image(tensor: torch.Tensor): # From official
    # ... (content of tensor_to_image from official utils.py) ...
    assert tensor.dim() == 3, "Input tensor should be 3-dimensional."
    assert tensor.dtype == torch.float32, "Input tensor should be float32."
    assert (
        tensor.min() >= 0 and tensor.max() <= 1
    ), "Input tensor should be in range [0, 1]."
    tensor = tensor.cpu()
    tensor = tensor * 255
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor.numpy().astype(np.uint8)
    image = Image.fromarray(tensor)
    return image


def concat_images(images: List[Image.Image], divider: int = 4, cols: int = 4): # From official
    # ... (content of concat_images from official utils.py) ...
    if not images: return None
    widths = [image.size[0] for image in images]
    heights = [image.size[1] for image in images]
    total_width = cols * max(widths) + divider * (cols - 1)
    rows = math.ceil(len(images) / cols)
    total_height = max(heights) * rows + divider * (rows - 1)
    concat_image = Image.new("RGB", (total_width, total_height), (0, 0, 0))
    x_offset, y_offset = 0, 0
    for i, image in enumerate(images):
        concat_image.paste(image, (x_offset, y_offset))
        if (i + 1) % cols == 0:
            x_offset = 0
            y_offset += image.size[1] + divider
        else:
            x_offset += image.size[0] + divider
    return concat_image


def read_prompt_file(prompt_file: str): # From official
    # ... (content of read_prompt_file from official utils.py) ...
    if prompt_file is not None and os.path.isfile(prompt_file):
        with open(prompt_file, "r") as sample_prompt_file:
            sample_prompts = sample_prompt_file.readlines()
            sample_prompts = [sample_prompt.strip() for sample_prompt in sample_prompts]
    else:
        sample_prompts = []
    return sample_prompts


def save_tensors_to_npz(tensors: torch.Tensor, paths: List[str]): # From official
    # ... (content of save_tensors_to_npz from official utils.py) ...
    assert len(tensors) == len(paths), "Length of tensors and paths should be the same!"
    for tensor, path in zip(tensors, paths):
        np.savez_compressed(path, latent=tensor.cpu().numpy())


def deepspeed_zero_init_disabled_context_manager(): # From official
    # ... (content of deepspeed_zero_init_disabled_context_manager from official utils.py) ...
    deepspeed_plugin = (
        AcceleratorState().deepspeed_plugin
        if accelerate.state.is_initialized()
        else None
    )
    if deepspeed_plugin is None:
        return []
    return [deepspeed_plugin.zero3_init_context_manager(enable=False)]


def is_xformers_available(): # From official
    # ... (content of is_xformers_available from official utils.py) ...
    try:
        import xformers
        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            print("xFormers 0.0.16 cannot be used ...")
        return True
    except ImportError:
        # raise ValueError("xformers is not available...") # Let's make it return False instead of raising error
        return False


def resize_and_crop(image, size): # From official AND your original utils.py
    w, h = image.size
    target_w, target_h = size
    if w / h < target_w / target_h:
        new_w = w
        new_h = w * target_h // target_w
    else:
        new_h = h
        new_w = h * target_w // target_h
    image = image.crop(
        ((w - new_w) // 2, (h - new_h) // 2, (w + new_w) // 2, (h + new_h) // 2)
    )
    image = image.resize(size, Image.LANCZOS)
    return image


def resize_and_padding(image, size): # From official AND your original utils.py
    w, h = image.size
    target_w, target_h = size
    if w / h < target_w / target_h:
        new_h = target_h
        new_w = w * target_h // h
    else:
        new_w = target_w
        new_h = h * target_w // w
    image = image.resize((new_w, new_h), Image.LANCZOS)
    padding = Image.new("RGB", size, (255, 255, 255))
    padding.paste(image, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return padding


def scan_files_in_dir(directory, postfix: Set[str] = None, progress_bar: tqdm = None) -> list: # From official
    # ... (content of scan_files_in_dir from official utils.py) ...
    file_list = []
    if progress_bar is None : progress_bar = tqdm(total=0, desc=f"Scanning", ncols=100)
    for entry in os.scandir(directory):
        if entry.is_file():
            if postfix is None or os.path.splitext(entry.path)[1] in postfix:
                file_list.append(entry.path) # Store path instead of entry object
                progress_bar.total += 1; progress_bar.update(1)
        elif entry.is_dir():
            file_list += scan_files_in_dir(entry.path, postfix=postfix, progress_bar=progress_bar)
    return file_list

# END OF MERGED utils.py
