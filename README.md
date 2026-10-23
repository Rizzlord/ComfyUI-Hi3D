# ComfyUI-Hi3D

ComfyUI nodes for [Hi3D v1.0](https://github.com/hi3d-official/Hi3Dv1.0), a high-resolution 3D geometry
refiner built on Sparc3D. It takes a reference image and a coarse mesh and returns a detailed mesh at
512³, 1024³ or 1536³.

Hi3D does not generate shapes from scratch. Give it a coarse mesh from any other generator, such as
Hunyuan3D, Trellis, TripoSG or a mesh you load from disk. Big structural errors in the coarse mesh
carry through to the result.

The nodes use the same `TRIMESH` socket (`trimesh.Trimesh`) as ComfyUI-Hunyuan3DWrapper and
ComfyUI-PhantyForge, so the result can go straight into their remesh, repair, UV and export nodes.

Powered by Hi3D 1.0.

## Nodes

| Node | In | Out |
| --- | --- | --- |
| **Hi3D Load Model** | model / DINOv2 / CLIP folders, `vram_mode`, `auto_download` | `HI3D_MODEL` |
| **Hi3D Refine** | `HI3D_MODEL`, `IMAGE`, `TRIMESH`, optional `MASK`, settings | `TRIMESH` |
| **Hi3D Export** | `TRIMESH`, filename prefix, glb/obj/ply/stl | file path (`STRING`, relative to `output/`) |

### Hi3D Refine settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `resolution` | 1024 | 512 runs 1 pass, 1024 runs 2 passes, 1536 runs 3 passes |
| `steps` | 25 | Scheduler updates per pass |
| `seed` | 0 | Sampler seed (CPU generator) |
| `scheduler` | heun | `heun` is the released 2nd-order sampler. `euler` makes about half the transformer calls |
| `guidance_strength` | 1.0 | Multiplier on the built-in image-guidance schedule. 0 turns guidance off |
| `guidance_cutoff` | 0.7 | Guidance only runs at model times above this value |
| `voxel_budget_scale` | 1.0 | Scales the sparse token budget per pass (80k/90k/110k). Lower uses less VRAM, higher keeps more detail |
| `tiled_decode` | on | Low-VRAM VAE decode (see below). Turn it off for the stock decode |
| `target_face_count` | 0 | Final face target. 0 means the default: 1M for 512/1024, 2M for 1536 |
| `component_face_ratio` | 0.005 | Small disconnected parts below this share of faces are dropped |
| `fill_holes_max_edges` | 10000 | Largest boundary loop that gets closed. 0 turns hole filling off |
| `restore_input_scale` | on | Puts the result on the input mesh's bounding box. Hi3D itself outputs at unit radius |
| `remove_background` | off | Runs rembg when there is no alpha and no mask |
| `force_offload` | on | Moves the weights back to the CPU after the run |
| `mask` / `invert_mask` | – | Foreground mask. Turn `invert_mask` on for the core LoadImage mask |
| `sigma_min`, `time_rescale_factor` | 0.002, 2.0 | Advanced flow-schedule parameters |

With the defaults, the output matches the upstream `Sparc3DPipeline`. Hi3D frames the object using
the image's alpha, so a cut-out PNG or a mask works best. Without alpha it uses the whole frame.

### VRAM modes (loader)

- `model_swap` (default): only the model that is currently running sits on the GPU. This mode is the one to use on 16 GB cards.
- `gpu_resident`: everything stays on the GPU. Fastest, but needs a lot of VRAM.
- `sequential_offload`: accelerate streams the weights layer by layer. It cuts weight memory, not decode memory.

Peak VRAM depends mostly on the VAE decode after each pass, not on the weights. Measured on an RTX 5080 (16 GB)
with `model_swap`, the default settings and a Hunyuan3D-2 mini coarse mesh:

| Resolution | Refine time | Peak VRAM (`tiled_decode` on) | `tiled_decode` off |
| --- | --- | --- | --- |
| 512 | ~40 s | 4.3 GB | 5.4 GB |
| 1024 | ~255 s | 9.2 GB | 12.6 GB; ran out of memory in one of two runs |
| 1536 | ~570 s | 11.2 GB | Out of memory unless `voxel_budget_scale` ≤ 0.6 |

With `sequential_offload`, 512 peaks at 2.5 GB and takes about 78 s. With `euler`, 512 takes about 21 s.

### Tiled decode

The VAE decoder's GroupNorm normalizes over every point of the shape at once, so cutting the shape into
spatial tiles (as image VAEs do) would change the statistics in each tile and leave seams. `tiled_decode`
processes the decoder in chunks of points instead:

- GroupNorm gathers its statistics across all chunks before normalizing.
- Sparse convolutions gather neighbours one chunk of rows at a time, without a full float32 copy of the input.
- Upsampling projects and prunes the 8 children chunk by chunk.

The math is the same as the stock decode; only fewer full-size copies are in memory at once. The output differs
from the stock decode only by the same tiny amount by which two stock decodes differ from each other (GPU atomic
adds): about 0.015% of points and a mean feature difference of 5e-4. On these shapes it was also faster than
the stock decode. The bundled `sparc3d` code is not modified; `tiled_decode.py` swaps in the chunked layers only
for the duration of a decode.

These memory fixes are always on, whatever `tiled_decode` is set to:

- The previous pass's decoded points are released before the next pass starts.
- The sparse-conv neighbour cache is dropped before mesh extraction.

**Coarse mesh from ComfyUI-Hunyuan3DWrapper:** set `force_offload` on *Hy3D Generate Mesh* to **False**.
That node's flag is inverted, so the default keeps Hunyuan (~3.4 GB) on the GPU while Hi3D runs.
Hi3D can't free it, because the wrapper isn't managed by ComfyUI.

## Install

The Hi3D inference code (the `sparc3d` package) is bundled unmodified in `Hi3D-1.0/`. You don't need to clone or install the Hi3D repository.

1. Put this folder in `ComfyUI/custom_nodes/`.
2. Make sure these are installed: `trimesh pyfqmr safetensors accelerate transformers huggingface_hub`
   (`rembg` is optional). You also need GPU kernels that match your torch/CUDA build: `spconv-cu12x`,
   `flash-attn` and `triton`. Install anything missing with `--no-deps`.
3. Models go in `ComfyUI/models/hi3d/`. With `auto_download` on, the loader fetches them on first use:
   ```
   models/hi3d/Hi3Dv1.0/                hi3d-official/Hi3Dv1.0 (8 runtime files, ~7.7 GB)
   models/hi3d/dinov2-large/            facebook/dinov2-large (config.json, model.safetensors)
   models/hi3d/clip-vit-large-patch14/  openai/clip-vit-large-patch14 (config.json, model.safetensors)
   ```

Requirements: a CUDA GPU with compute capability 8.0 or higher. There is no CPU path. Tested with
Python 3.12, torch 2.9.1+cu128, spconv-cu128 2.4.0, flash-attn 2.8.3 and transformers 5.5 on an
RTX 5080 (16 GB).

## License

Hi3D code and weights are covered by the Hi3D Open License Agreement v1.0 (see `LICENSE` and `NOTICE`).
Commercial integrations must display "Powered by Hi3D 1.0". The bundled code also keeps its Apache-2.0 and
MIT portions. Their terms are in `THIRD_PARTY_LICENSES.md`, `AUTHORS` and
`Hi3D-1.0/sparc3d/ops/flexicubes/LICENSE.txt`.
