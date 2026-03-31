#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import torch

from pi3.models.pi3x import Pi3X
from pi3.utils.geometry import depth_edge, recover_intrinsic_from_rays_d


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Pi3X inference in Docker and export comprehensive outputs."
    )
    parser.add_argument(
        "--image_folder", type=str, required=True, help="Input image folder."
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional local checkpoint path inside container (.safetensors or .pth).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Inference device (default: auto).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Sampling interval over sorted images (default: 1).",
    )
    parser.add_argument(
        "--max_views",
        type=int,
        default=None,
        help="Optional cap on number of views after interval sampling.",
    )
    parser.add_argument(
        "--pixel_limit",
        type=int,
        default=255000,
        help="Maximum target pixels used for resizing (default: 255000).",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.1,
        help="Confidence sigmoid threshold for masking (default: 0.1).",
    )
    parser.add_argument(
        "--edge_rtol",
        type=float,
        default=0.03,
        help="Relative depth edge threshold (default: 0.03).",
    )
    return parser.parse_args()


def resolve_device(mode: str) -> torch.device:
    if mode == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(mode)


def normalize_depth_for_viz(depth: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    depth_viz = np.zeros(depth.shape, dtype=np.uint8)
    valid_depth = depth[valid_mask]
    if valid_depth.size == 0:
        return depth_viz

    near = np.percentile(valid_depth, 2)
    far = np.percentile(valid_depth, 98)
    if far <= near:
        near = float(valid_depth.min())
        far = float(valid_depth.max())
    if far <= near:
        depth_viz[valid_mask] = 255
        return depth_viz

    scaled = np.clip((depth - near) / (far - near), 0.0, 1.0)
    depth_viz[valid_mask] = (scaled[valid_mask] * 255.0).astype(np.uint8)
    return depth_viz


def write_ply(points_xyz: np.ndarray, points_rgb: np.ndarray, out_path: Path) -> None:
    if points_xyz.shape[0] == 0:
        raise ValueError("No points to save in PLY.")

    colors = np.clip(points_rgb * 255.0, 0, 255).astype(np.uint8)
    normals = np.zeros_like(points_xyz, dtype=np.float32)

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    elements = np.empty(points_xyz.shape[0], dtype=dtype)
    attributes = np.concatenate(
        (points_xyz.astype(np.float32), normals, colors), axis=1
    )
    elements[:] = list(map(tuple, attributes))
    ply_data = PlyData([PlyElement.describe(elements, "vertex")])
    ply_data.write(str(out_path))


def collect_image_paths(image_folder: Path) -> list[Path]:
    image_paths = sorted(
        path
        for path in image_folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    return image_paths


def load_images_as_tensor(image_paths: list[Path], pixel_limit: int) -> torch.Tensor:
    if not image_paths:
        return torch.empty(0)

    pil_images: list[Image.Image] = []
    for path in image_paths:
        pil_images.append(Image.open(path).convert("RGB"))

    first_img = pil_images[0]
    width_orig, height_orig = first_img.size
    scale = (
        math.sqrt(pixel_limit / (width_orig * height_orig))
        if width_orig * height_orig > 0
        else 1.0
    )
    width_target = width_orig * scale
    height_target = height_orig * scale

    k = round(width_target / 14)
    m = round(height_target / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / max(m, 1) > width_target / max(height_target, 1):
            k -= 1
        else:
            m -= 1

    target_w = max(1, k) * 14
    target_h = max(1, m) * 14

    tensors: list[torch.Tensor] = []
    for img in pil_images:
        resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())

    return torch.stack(tensors, dim=0)


def save_rgb_image(rgb_float: np.ndarray, out_path: Path) -> None:
    rgb_uint8 = np.clip(rgb_float * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(rgb_uint8, mode="RGB").save(out_path)


def main() -> None:
    args = parse_args()
    if args.interval < 1:
        raise ValueError("--interval must be >= 1")
    if args.max_views is not None and args.max_views < 1:
        raise ValueError("--max_views must be >= 1 when provided")

    image_folder = Path(args.image_folder).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_folder.is_dir():
        raise ValueError(f"Input image folder not found: {image_folder}")

    image_paths = collect_image_paths(image_folder)
    if len(image_paths) == 0:
        raise ValueError(
            f"No images found in {image_folder} (searched recursively for: {sorted(SUPPORTED_IMAGE_SUFFIXES)})"
        )

    image_paths = image_paths[:: args.interval]
    if args.max_views is not None:
        image_paths = image_paths[: args.max_views]
    if len(image_paths) == 0:
        raise ValueError("No images selected after applying --interval/--max_views")

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Input views: {len(image_paths)}")

    model_name = "yyfz233/Pi3X"
    if args.ckpt is not None:
        print(f"Loading model with local checkpoint: {args.ckpt}")
        model = Pi3X(use_multimodal=False).eval()
        if args.ckpt.endswith(".safetensors"):
            from safetensors.torch import load_file

            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(weight, strict=False)
    else:
        print(f"Downloading/loading pretrained model: {model_name}")
        model = Pi3X.from_pretrained(model_name).eval()
    model.disable_multimodal()
    model = model.to(device)

    imgs = load_images_as_tensor(image_paths, pixel_limit=args.pixel_limit)
    if imgs.shape[0] == 0:
        raise ValueError("No images loaded for inference")
    imgs = imgs.to(device)

    print("Running inference...")
    with torch.no_grad():
        if device.type == "cuda":
            capability = torch.cuda.get_device_capability(device=device)
            amp_dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                res = model(imgs[None])
        else:
            res = model(imgs[None])
    print("Inference complete")

    masks = torch.sigmoid(res["conf"][..., 0]) > args.conf_threshold
    non_edge = ~depth_edge(res["local_points"][..., 2], rtol=args.edge_rtol)
    masks = torch.logical_and(masks, non_edge)[0]

    points = res["points"][0]
    local_points = res["local_points"][0]
    rays = torch.nn.functional.normalize(res["local_points"], dim=-1)
    intrinsics = recover_intrinsic_from_rays_d(rays, force_center_principal_point=True)[
        0
    ]
    extrinsics = res["camera_poses"][0]

    depth_npy_dir = output_dir / "depth_npy"
    depth_png_dir = output_dir / "depth_png"
    intrinsics_dir = output_dir / "intrinsics"
    extrinsics_dir = output_dir / "extrinsics"
    images_dir = output_dir / "images"
    mask_dir = output_dir / "mask_npy"
    for directory in [
        depth_npy_dir,
        depth_png_dir,
        intrinsics_dir,
        extrinsics_dir,
        images_dir,
        mask_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    imgs_cpu = imgs.detach().cpu()
    points_cpu = points.detach().cpu()
    local_points_cpu = local_points.detach().cpu()
    masks_cpu = masks.detach().cpu()
    intrinsics_cpu = intrinsics.detach().cpu()
    extrinsics_cpu = extrinsics.detach().cpu()

    for idx in range(imgs_cpu.shape[0]):
        stem = f"view_{idx:04d}"
        depth = local_points_cpu[idx, ..., 2].numpy().astype(np.float32)
        valid_mask = masks_cpu[idx].numpy().astype(bool)
        depth_viz = normalize_depth_for_viz(depth, valid_mask)

        np.save(depth_npy_dir / f"{stem}.npy", depth)
        np.save(mask_dir / f"{stem}.npy", valid_mask)
        np.save(
            intrinsics_dir / f"{stem}.npy",
            intrinsics_cpu[idx].numpy().astype(np.float32),
        )
        np.save(
            extrinsics_dir / f"{stem}.npy",
            extrinsics_cpu[idx].numpy().astype(np.float32),
        )
        Image.fromarray(depth_viz, mode="L").save(depth_png_dir / f"{stem}.png")

        rgb = imgs_cpu[idx].permute(1, 2, 0).numpy().astype(np.float32)
        save_rgb_image(rgb, images_dir / f"{stem}.png")

    flat_points = points_cpu[masks_cpu]
    flat_colors = imgs_cpu.permute(0, 2, 3, 1)[masks_cpu]
    if flat_points.shape[0] == 0:
        flat_points = points_cpu.reshape(-1, 3)
        flat_colors = imgs_cpu.permute(0, 2, 3, 1).reshape(-1, 3)

    ply_path = output_dir / "reconstruction.ply"
    write_ply(flat_points.numpy(), flat_colors.numpy(), ply_path)

    metadata = {
        "model": model_name if args.ckpt is None else "local_checkpoint",
        "checkpoint": args.ckpt,
        "device": str(device),
        "num_input_files": len(image_paths),
        "image_folder": str(image_folder),
        "output_dir": str(output_dir),
        "interval": args.interval,
        "max_views": args.max_views,
        "pixel_limit": args.pixel_limit,
        "conf_threshold": args.conf_threshold,
        "edge_rtol": args.edge_rtol,
        "points_in_ply": int(flat_points.shape[0]),
        "image_shape": {
            "num_views": int(imgs_cpu.shape[0]),
            "height": int(imgs_cpu.shape[2]),
            "width": int(imgs_cpu.shape[3]),
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved comprehensive outputs to: {output_dir}")
    print(f"Saved PLY: {ply_path}")


if __name__ == "__main__":
    main()
