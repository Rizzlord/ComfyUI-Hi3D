import os
import time
import logging

import numpy as np
import torch
import trimesh
from PIL import Image

import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar

log = logging.getLogger("Hi3D")

try:
    from .hi3d_pipeline import ComfyHi3DPipeline, RefineOptions, SCHEDULERS, VRAM_MODES
    HI3D_IMPORT_ERROR = None
except Exception as e:  # keep ComfyUI booting; the loader reports the problem
    ComfyHi3DPipeline = RefineOptions = None
    SCHEDULERS = {"heun": None, "euler": None}
    VRAM_MODES = ("model_swap", "gpu_resident", "sequential_offload")
    HI3D_IMPORT_ERROR = e
    log.error(f"[Hi3D] failed to import sparc3d: {e}")

hi3d_models_dir = os.path.join(folder_paths.models_dir, "hi3d")
os.makedirs(hi3d_models_dir, exist_ok=True)
folder_paths.add_model_folder_path("hi3d", hi3d_models_dir)

HI3D_REPO_FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "vae/config.json",
    "vae/model.safetensors",
    "transformer_512/config.json",
    "transformer_512/model.safetensors",
    "transformer_1024/config.json",
    "transformer_1024/model.safetensors",
]
# Default folder name -> (Hugging Face repo, files fetched when auto-downloading)
DOWNLOADS = {
    "Hi3Dv1.0": ("hi3d-official/Hi3Dv1.0", HI3D_REPO_FILES),
    "dinov2-large": ("facebook/dinov2-large", ["config.json", "model.safetensors"]),
    "clip-vit-large-patch14": ("openai/clip-vit-large-patch14", ["config.json", "model.safetensors"]),
}


def _hi3d_roots():
    return [p for p in folder_paths.get_folder_paths("hi3d") if os.path.isdir(p)]


def _list_subfolders(marker, defaults, exclude_marker=None):
    found = []
    for root in _hi3d_roots():
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if name.startswith(".") or not os.path.isfile(os.path.join(path, marker)):
                continue
            if exclude_marker and os.path.isfile(os.path.join(path, exclude_marker)):
                continue
            if name not in found:
                found.append(name)
    # Defaults stay selectable so auto_download can fetch them on first use.
    return found + [d for d in defaults if d not in found]


def _resolve_folder(name, marker, auto_download):
    # A known folder only counts once every file is there, so an interrupted download resumes.
    required = DOWNLOADS[name][1] if name in DOWNLOADS else [marker]
    for root in _hi3d_roots():
        path = os.path.join(root, name)
        if all(os.path.isfile(os.path.join(path, f)) for f in required):
            return os.path.abspath(path)
    target = os.path.join(hi3d_models_dir, name)
    if name not in DOWNLOADS:
        raise FileNotFoundError(f"[Hi3D] model folder not found: {target}")
    if not auto_download:
        raise FileNotFoundError(f"[Hi3D] {target} is missing; enable auto_download or download {DOWNLOADS[name][0]} there")
    from huggingface_hub import snapshot_download
    repo_id, files = DOWNLOADS[name]
    log.info(f"[Hi3D] downloading {repo_id} to {target}")
    snapshot_download(repo_id=repo_id, local_dir=target, allow_patterns=files)
    return os.path.abspath(target)


def _comfy_image_to_pil(image, mask=None, invert_mask=False, remove_background=False):
    """First IMAGE of a batch to a PIL image, RGBA whenever an alpha is available."""
    frame = image[0].detach().cpu().float().numpy()
    rgb = Image.fromarray((np.clip(frame[..., :3], 0, 1) * 255).round().astype(np.uint8), "RGB")
    alpha = None
    if mask is not None:
        m = mask[0] if mask.ndim == 3 else mask
        m = m.detach().cpu().float().numpy()
        if invert_mask:
            m = 1.0 - m
        alpha = Image.fromarray((np.clip(m, 0, 1) * 255).round().astype(np.uint8), "L")
        if alpha.size != rgb.size:
            alpha = alpha.resize(rgb.size, Image.BILINEAR)
    elif frame.shape[-1] == 4:
        alpha = Image.fromarray((np.clip(frame[..., 3], 0, 1) * 255).round().astype(np.uint8), "L")

    if alpha is not None:
        # Hi3D crops to the bounding box of any non-zero alpha, so drop faint mask
        # noise; an all-transparent mask (e.g. LoadImage on a JPG) means "no alpha".
        alpha = alpha.point(lambda v: 0 if v < 8 else v)
        if alpha.getbbox() is None:
            log.warning("[Hi3D] mask is empty, ignoring it (try toggling invert_mask)")
            alpha = None
    if alpha is not None:
        rgb.putalpha(alpha)
        return rgb
    if remove_background:
        from sparc3d.utils.image import remove_background as rembg_remove
        return rembg_remove(rgb)
    return rgb


def _as_trimesh(mesh):
    if isinstance(mesh, (list, tuple)):
        mesh = trimesh.util.concatenate([m.to_geometry() if isinstance(m, trimesh.Scene) else m for m in mesh])
    if not isinstance(mesh, (trimesh.Trimesh, trimesh.Scene)):
        raise TypeError(f"[Hi3D] expected a trimesh.Trimesh, got {type(mesh).__name__}")
    return mesh


def _match_bounds(mesh, reference):
    """Map ``mesh`` onto the bounding-box centre and largest extent of ``reference``."""
    ref_lo, ref_hi = reference.bounds
    lo, hi = mesh.bounds
    scale = float((ref_hi - ref_lo).max()) / max(float((hi - lo).max()), 1e-12)
    vertices = (mesh.vertices - (lo + hi) / 2.0) * scale + (ref_lo + ref_hi) / 2.0
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)


class Hi3DLoadModel:
    _cache = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (_list_subfolders("model_index.json", ["Hi3Dv1.0"]), {"tooltip": "Hi3D model folder inside 'ComfyUI/models/hi3d'"}),
                "dino_model": (_list_subfolders("config.json", ["dinov2-large"], "model_index.json"), {"default": "dinov2-large", "tooltip": "facebook/dinov2-large image encoder folder inside 'ComfyUI/models/hi3d'"}),
                "clip_model": (_list_subfolders("config.json", ["clip-vit-large-patch14"], "model_index.json"), {"default": "clip-vit-large-patch14", "tooltip": "openai/clip-vit-large-patch14 image encoder folder inside 'ComfyUI/models/hi3d'"}),
                "vram_mode": (list(VRAM_MODES), {"default": "model_swap", "tooltip": "model_swap: only the running model sits on the GPU (fits 16 GB cards).\ngpu_resident: everything stays on the GPU (fastest, needs lots of VRAM).\nsequential_offload: accelerate layer streaming (lowest VRAM, very slow)."}),
                "auto_download": ("BOOLEAN", {"default": True, "tooltip": "Download missing default models from Hugging Face into 'ComfyUI/models/hi3d'"}),
            },
        }

    RETURN_TYPES = ("HI3D_MODEL",)
    RETURN_NAMES = ("hi3d_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "Hi3D"

    def loadmodel(self, model, dino_model, clip_model, vram_mode, auto_download):
        if HI3D_IMPORT_ERROR is not None:
            raise RuntimeError(f"[Hi3D] sparc3d could not be imported: {HI3D_IMPORT_ERROR}") from HI3D_IMPORT_ERROR
        device = mm.get_torch_device()
        if device.type != "cuda":
            raise RuntimeError("[Hi3D] Hi3D needs a CUDA GPU (flash-attn, spconv and Triton kernels)")

        model_dir = _resolve_folder(model, "model_index.json", auto_download)
        dino_dir = _resolve_folder(dino_model, "config.json", auto_download)
        clip_dir = _resolve_folder(clip_model, "config.json", auto_download)

        key = (model_dir, dino_dir, clip_dir, vram_mode, str(device))
        if key in self._cache:
            return (self._cache[key],)
        for old in self._cache.values():
            old.offload_all()
        self._cache.clear()
        mm.soft_empty_cache()

        log.info(f"[Hi3D] loading {model_dir} ({vram_mode})")
        pipeline = ComfyHi3DPipeline.from_pretrained(
            model_dir,
            local_files_only=True,
            dino_model_id=dino_dir,
            clip_model_id=clip_dir,
        )
        pipeline.setup_device(device, vram_mode)
        self._cache[key] = pipeline
        return (pipeline,)


class Hi3DRefine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hi3d_model": ("HI3D_MODEL",),
                "image": ("IMAGE", {"tooltip": "Reference image of the object. A cut-out with alpha (or a mask) gives the best framing"}),
                "trimesh": ("TRIMESH", {"tooltip": "Coarse mesh to refine (Hunyuan3D, Trellis, TripoSG, a loaded mesh, ...)"}),
                "resolution": ([512, 1024, 1536], {"default": 1024, "tooltip": "Sparse extraction grid. 512 = 1 pass, 1024 = 2 passes, 1536 = 3 passes"}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200, "tooltip": "Scheduler updates per pass"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "fixed"}),
                "scheduler": (list(SCHEDULERS), {"default": "heun", "tooltip": "heun: released 2nd-order sampler. euler: 1st order, about half the transformer calls"}),
                "guidance_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05, "tooltip": "Multiplier on the built-in image guidance schedule. 1.0 = as released, 0 = no guidance"}),
                "guidance_cutoff": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Guidance runs only at model times above this value (released: 0.7). Lower = guided in more steps, slower"}),
                "voxel_budget_scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 1.5, "step": 0.05, "tooltip": "Scales the sparse token budget of each pass (80k/90k/110k). Lower saves VRAM and time, higher keeps more detail"}),
                "tiled_decode": ("BOOLEAN", {"default": True, "tooltip": "Decode the VAE in chunks of points with global GroupNorm statistics. Much lower VRAM and faster; matches the normal decode within its run-to-run noise"}),
                "reduce_faces": ("BOOLEAN", {"default": True, "tooltip": "Simplify the final mesh to target_face_count. Off keeps the full extracted geometry (several million faces at 1024/1536)"}),
                "target_face_count": ("INT", {"default": 0, "min": 0, "max": 10000000, "step": 1000, "tooltip": "Simplification target when reduce_faces is on. 0 = default (1M for 512/1024, 2M for 1536)"}),
                "remove_floaters": ("BOOLEAN", {"default": True, "tooltip": "Remove small disconnected parts, before and after simplification. The largest part is always kept"}),
                "floater_face_ratio": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Parts with less than this share of all faces count as floaters. 1.0 keeps only the largest part"}),
                "fill_holes_max_edges": ("INT", {"default": 10000, "min": 0, "max": 100000, "tooltip": "Close boundary loops up to this many edges after each extraction. 0 disables hole filling"}),
                "restore_input_scale": ("BOOLEAN", {"default": True, "tooltip": "Place the result on the input mesh's bounding box. Off = Hi3D's unit-radius output"}),
                "remove_background": ("BOOLEAN", {"default": False, "tooltip": "Run rembg when the image has no alpha and no mask is connected"}),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Move all Hi3D weights back to the CPU when done"}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Foreground mask (white = object)"}),
                "invert_mask": ("BOOLEAN", {"default": False, "tooltip": "Enable for the MASK output of the core LoadImage node, which is inverted (white = transparent)"}),
                "sigma_min": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 0.5, "step": 0.001, "tooltip": "Terminal noise level of the flow schedule"}),
                "time_rescale_factor": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "Rational time rescale applied to the schedule before the token-count warp"}),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "refine"
    CATEGORY = "Hi3D"

    def refine(self, hi3d_model, image, trimesh, resolution, steps, seed, scheduler, guidance_strength,
               guidance_cutoff, voxel_budget_scale, tiled_decode, reduce_faces, target_face_count,
               remove_floaters, floater_face_ratio, fill_holes_max_edges, restore_input_scale, remove_background, force_offload,
               mask=None, invert_mask=False, sigma_min=0.002, time_rescale_factor=2.0):
        pipeline = hi3d_model
        coarse = _as_trimesh(trimesh)
        resolution = int(resolution)
        pil_image = _comfy_image_to_pil(image, mask, invert_mask, remove_background)
        if pil_image.mode != "RGBA":
            log.warning("[Hi3D] image has no alpha: the whole frame is used for conditioning")

        pipeline.configure(RefineOptions(
            scheduler=scheduler,
            sigma_min=sigma_min,
            time_rescale_factor=time_rescale_factor,
            guidance_strength=guidance_strength,
            guidance_cutoff=guidance_cutoff,
            voxel_budget_scale=voxel_budget_scale,
            tiled_decode=tiled_decode,
            reduce_faces=reduce_faces,
            target_face_count=target_face_count,
            remove_floaters=remove_floaters,
            floater_face_ratio=floater_face_ratio,
            fill_holes_max_edges=fill_holes_max_edges,
        ))

        num_passes = (512, 1024, 1536).index(resolution) + 1
        pbar = ProgressBar(num_passes * steps)

        def on_step(event):
            mm.throw_exception_if_processing_interrupted()
            pbar.update_absolute(event.pass_index * event.num_inference_steps + event.step, num_passes * steps)
            if event.step == event.num_inference_steps:
                log.info(f"[Hi3D] pass {event.pass_index + 1}/{event.num_passes} ({event.structure_resolution}^3) "
                         f"sampled at {time.perf_counter() - start:.1f}s, peak VRAM {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

        mm.unload_all_models()
        mm.soft_empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        pipeline.progress_callback = on_step
        try:
            pipeline.prepare_for_run()
            result = pipeline(pil_image, coarse, resolution=resolution, num_inference_steps=steps, seed=int(seed))
        except BaseException:
            pipeline.offload_all()
            raise
        finally:
            pipeline.progress_callback = None
        if force_offload:
            pipeline.offload_all()
        mm.soft_empty_cache()

        mesh = result.mesh
        if restore_input_scale:
            mesh = _match_bounds(mesh, coarse)
        log.info(f"[Hi3D] {resolution} refine done in {time.perf_counter() - start:.1f}s: "
                 f"{len(mesh.vertices)} vertices, {len(mesh.faces)} faces, "
                 f"peak VRAM {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        return (mesh,)


class Hi3DExport:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "filename_prefix": ("STRING", {"default": "3D/Hi3D"}),
                "file_format": (["glb", "obj", "ply", "stl"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "export"
    CATEGORY = "Hi3D"
    OUTPUT_NODE = True

    def export(self, trimesh, filename_prefix, file_format):
        mesh = _as_trimesh(trimesh)
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        os.makedirs(full_output_folder, exist_ok=True)
        name = f"{filename}_{counter:05}_.{file_format}"
        mesh.export(os.path.join(full_output_folder, name), file_type=file_format)
        relative_path = os.path.join(subfolder, name)
        log.info(f"[Hi3D] saved {relative_path}")
        return {"ui": {"3d": [{"filename": name, "subfolder": subfolder, "type": "output"}]}, "result": (relative_path,)}


NODE_CLASS_MAPPINGS = {
    "Hi3DLoadModel": Hi3DLoadModel,
    "Hi3DRefine": Hi3DRefine,
    "Hi3DExport": Hi3DExport,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Hi3DLoadModel": "Hi3D Load Model",
    "Hi3DRefine": "Hi3D Refine",
    "Hi3DExport": "Hi3D Export",
}
