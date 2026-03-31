#!/usr/bin/env python3

import argparse
import shlex
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and run Pi3 Docker inference with host path mapping."
    )
    parser.add_argument(
        "--image-folder", type=str, required=True, help="Host input image folder."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Host output folder."
    )
    parser.add_argument(
        "--image-name", type=str, default="pi3:cuda12.8", help="Docker image name/tag."
    )
    parser.add_argument(
        "--dockerfile", type=str, default="Dockerfile", help="Path to Dockerfile."
    )
    parser.add_argument(
        "--build",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build docker image before run (default: true).",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="all",
        help="Value passed to --gpus for docker run (default: all).",
    )
    parser.add_argument(
        "--runtime",
        type=str,
        default=None,
        help="Optional docker runtime override (for example: runc).",
    )
    parser.add_argument(
        "--hf-cache",
        type=str,
        default=str(Path.home() / ".cache" / "huggingface"),
        help="Host Hugging Face cache directory to mount.",
    )
    parser.add_argument(
        "--torch-cache",
        type=str,
        default=str(Path.home() / ".cache" / "torch"),
        help="Host Torch cache directory to mount.",
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="Optional host checkpoint file path."
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Container inference device mode.",
    )
    parser.add_argument(
        "--interval", type=int, default=1, help="Image sampling interval."
    )
    parser.add_argument(
        "--max-views",
        type=int,
        default=None,
        help="Optional cap on number of input views.",
    )
    parser.add_argument(
        "--pixel-limit", type=int, default=255000, help="Max target pixels."
    )
    parser.add_argument(
        "--conf-threshold", type=float, default=0.1, help="Confidence threshold."
    )
    parser.add_argument(
        "--edge-rtol", type=float, default=0.03, help="Depth edge relative threshold."
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print commands and exit without execution.",
    )
    return parser.parse_args()


def run_command(cmd: list[str], dry_run: bool, cwd: Path | None = None) -> None:
    print(shlex.join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=cwd)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    dockerfile_path = (repo_root / args.dockerfile).resolve()

    input_dir = Path(args.image_folder).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    hf_cache = Path(args.hf_cache).expanduser().resolve()
    torch_cache = Path(args.torch_cache).expanduser().resolve()

    if not input_dir.is_dir():
        raise ValueError(f"Input folder not found: {input_dir}")
    if not dockerfile_path.is_file():
        raise ValueError(f"Dockerfile not found: {dockerfile_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    hf_cache.mkdir(parents=True, exist_ok=True)
    torch_cache.mkdir(parents=True, exist_ok=True)

    ckpt_host_path = None
    ckpt_container_path = None
    if args.ckpt is not None:
        ckpt_host_path = Path(args.ckpt).expanduser().resolve()
        if not ckpt_host_path.is_file():
            raise ValueError(f"Checkpoint file not found: {ckpt_host_path}")
        ckpt_container_path = f"/ckpt/{ckpt_host_path.name}"

    if args.build:
        build_cmd = [
            "docker",
            "build",
            "-t",
            args.image_name,
            "-f",
            str(dockerfile_path),
            ".",
        ]
        run_command(build_cmd, args.dry_run, cwd=repo_root)

    run_cmd = ["docker", "run", "--rm"]
    if args.runtime is not None:
        run_cmd.extend(["--runtime", args.runtime])
    if args.gpus:
        run_cmd.extend(["--gpus", args.gpus])

    run_cmd.extend(
        [
            "-v",
            f"{input_dir}:/input:ro",
            "-v",
            f"{output_dir}:/output",
            "-v",
            f"{hf_cache}:/root/.cache/huggingface",
            "-v",
            f"{torch_cache}:/root/.cache/torch",
        ]
    )

    if ckpt_host_path is not None:
        run_cmd.extend(["-v", f"{ckpt_host_path.parent}:/ckpt:ro"])

    run_cmd.append(args.image_name)
    run_cmd.extend(
        [
            "--image_folder",
            "/input",
            "--output_dir",
            "/output",
            "--device",
            args.device,
            "--interval",
            str(args.interval),
            "--pixel_limit",
            str(args.pixel_limit),
            "--conf_threshold",
            str(args.conf_threshold),
            "--edge_rtol",
            str(args.edge_rtol),
        ]
    )
    if args.max_views is not None:
        run_cmd.extend(["--max_views", str(args.max_views)])
    if ckpt_container_path is not None:
        run_cmd.extend(["--ckpt", ckpt_container_path])

    run_command(run_cmd, args.dry_run)


if __name__ == "__main__":
    main()
