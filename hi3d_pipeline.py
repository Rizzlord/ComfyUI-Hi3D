"""Sparc3DPipeline subclass adding per-call settings and VRAM residency control.

The bundled upstream ``sparc3d`` package is unmodified: every knob is applied by
overriding one of the pipeline's ``self.``-dispatched stages, and the defaults in
:class:`RefineOptions` reproduce the upstream behaviour exactly.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from dataclasses import dataclass

import torch

# The upstream package is bundled in Hi3D-1.0/ and imports itself as top-level ``sparc3d``.
hi3d_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Hi3D-1.0")
if hi3d_lib_path not in sys.path:
    sys.path.insert(0, hi3d_lib_path)

from sparc3d import Sparc3DPipeline
from sparc3d.schedulers.flow_match_euler import FlowMatchEulerScheduler
from sparc3d.schedulers.flow_match_heun import FlowMatchHeunScheduler

from .tiled_decode import tiled_decode

log = logging.getLogger("Hi3D")

VRAM_MODES = ("model_swap", "gpu_resident", "sequential_offload")
SCHEDULERS = {"heun": FlowMatchHeunScheduler, "euler": FlowMatchEulerScheduler}


@dataclass
class RefineOptions:
    """Settings read by the overridden pipeline stages during one call."""

    scheduler: str = "heun"
    sigma_min: float = 0.002
    time_rescale_factor: float = 2.0
    guidance_strength: float = 1.0
    guidance_cutoff: float = 0.7
    voxel_budget_scale: float = 1.0
    target_face_count: int = 0
    component_face_ratio: float = 0.005
    fill_holes_max_edges: int = 10000
    tiled_decode: bool = True


def _module_device(module: torch.nn.Module) -> torch.device:
    for tensor in module.parameters():
        return tensor.device
    for tensor in module.buffers():
        return tensor.device
    return torch.device("cpu")


class ComfyHi3DPipeline(Sparc3DPipeline):
    """Hi3D pipeline that ComfyUI can tune per call and fit on smaller GPUs."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.options = RefineOptions()
        self.vram_mode = "gpu_resident"
        self.execution_device = torch.device("cpu")
        self._swap_device: torch.device | None = None

    # ------------------------------------------------------------------ devices

    def setup_device(self, device: torch.device, vram_mode: str) -> "ComfyHi3DPipeline":
        """Choose how weights live on ``device``; call once, right after loading."""
        if vram_mode not in VRAM_MODES:
            raise ValueError(f"vram_mode must be one of {VRAM_MODES}, got {vram_mode!r}")
        device = torch.device(device)
        if vram_mode == "sequential_offload":
            self.enable_sequential_cpu_offload(device)
        elif vram_mode == "model_swap":
            # Tensors are created on the GPU while weights stay on the CPU until the
            # module that owns them runs; see _activate.
            self._device = device
            self._swap_device = device
            for module in (self.conditioner, self.transformer_512):
                module.register_forward_pre_hook(self._swap_hook)
        self.vram_mode = vram_mode
        self.execution_device = device
        return self

    @property
    def transformer_1024(self):
        if self._transformer_1024 is None and self._swap_device is not None:
            if self._transformer_1024_loader is None:
                raise RuntimeError("the transformer_1024 loader has already been consumed")
            log.info("Loading transformer_1024 on CPU")
            model = self._transformer_1024_loader()
            self._transformer_1024_loader = None
            model.register_forward_pre_hook(self._swap_hook)
            self._transformer_1024 = model
        return Sparc3DPipeline.transformer_1024.fget(self)

    def _swappable(self) -> list[torch.nn.Module]:
        modules = [self.conditioner, self.transformer_512, self.vae]
        if self._transformer_1024 is not None:
            modules.append(self._transformer_1024)
        return modules

    def _activate(self, module: torch.nn.Module | None) -> None:
        """Keep only ``module`` on the GPU (none at all when ``module`` is None)."""
        if self._swap_device is None:
            return
        moved = False
        for other in self._swappable():
            if other is not module and _module_device(other).type != "cpu":
                other.to("cpu")
                moved = True
        if moved:
            torch.cuda.empty_cache()
        if module is not None and _module_device(module) != self._swap_device:
            module.to(self._swap_device)

    def _swap_hook(self, module, args) -> None:
        self._activate(module)

    def prepare_for_run(self) -> None:
        if self.vram_mode == "gpu_resident" and self._device != self.execution_device:
            self.to(self.execution_device)

    def offload_all(self) -> None:
        """Move every materialized module to the CPU."""
        if self.vram_mode == "model_swap":
            self._activate(None)
        elif self.vram_mode == "gpu_resident" and self._device.type != "cpu":
            self.to("cpu")
        torch.cuda.empty_cache()

    # --------------------------------------------------------------- settings

    def configure(self, options: RefineOptions) -> None:
        if options.scheduler not in SCHEDULERS:
            raise ValueError(f"scheduler must be one of {tuple(SCHEDULERS)}")
        self.options = options
        self.scheduler = SCHEDULERS[options.scheduler](
            sigma_min=options.sigma_min,
            time_rescale_factor=options.time_rescale_factor,
        )

    def _guided_velocity(self, transformer, sample, timestep, condition, negative_condition):
        # Upstream: scale 4 at t >= 1, 10 t^2 on [0.7, 1), no guidance below 0.7.
        # The cutoff moves the 0.7 boundary and strength scales the excess over 1.
        options = self.options
        model_timestep = (1000.0 * timestep).reshape(1)
        positive = transformer(sample, model_timestep, condition)
        time = float(timestep)
        if time < options.guidance_cutoff or options.guidance_strength == 0.0:
            return positive.feats
        negative = transformer(sample, model_timestep, negative_condition)
        scale = 4.0 if time >= 1.0 else max(1.0, 10.0 * time**2)
        scale = 1.0 + options.guidance_strength * (scale - 1.0)
        return negative.feats + scale * (positive.feats - negative.feats)

    def _adaptive_voxelize(self, mesh, structure_resolution, token_budget):
        budget = max(1000, int(round(token_budget * self.options.voxel_budget_scale)))
        return super()._adaptive_voxelize(mesh, structure_resolution, budget)

    def _decode_latents(self, latents, structure_resolution):
        self._activate(self.vae)
        with tiled_decode(self.vae) if self.options.tiled_decode else contextlib.nullcontext():
            return super()._decode_latents(latents, structure_resolution)

    def _extract_mesh(self, decoded):
        import trimesh

        from sparc3d.sparcubes.extract import point_to_mesh
        from sparc3d.sparcubes.fill_hole import fill_mesh_holes

        # Isosurface extraction needs no weights, so give it the whole GPU. It also
        # never reads the convolution view, whose cached neighbour pairs are large.
        self._activate(None)
        decoded.pop("sparse_conv_feat", None)
        torch.cuda.empty_cache()
        mesh = point_to_mesh(decoded)
        # The caller keeps `decoded` alive into the next pass; release its tensors now.
        decoded.clear()
        if self.options.fill_holes_max_edges > 0:
            mesh = fill_mesh_holes(mesh, length_thresh=self.options.fill_holes_max_edges)
        if not mesh.success:
            raise RuntimeError("mesh extraction produced an empty isosurface")
        return trimesh.Trimesh(
            vertices=mesh.vertices.detach().cpu().numpy(),
            faces=mesh.faces.detach().cpu().numpy(),
            process=False,
        )

    def _postprocess_mesh(self, mesh, *, target_face_count):
        from sparc3d.sparcubes.postprocess import postprocess_mesh

        return postprocess_mesh(
            mesh,
            target_face_count=self.options.target_face_count or target_face_count,
            component_face_ratio=self.options.component_face_ratio,
        )
