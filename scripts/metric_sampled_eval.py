from argparse import ArgumentParser
from pathlib import Path
import sys

from PIL import Image

try:
    from .evaluation_common import (
        atomic_write_json,
        collect_indexed_images,
        require_primary_coverage,
        resolve_torch_device,
        write_status_error,
    )
except ImportError:
    from evaluation_common import (
        atomic_write_json,
        collect_indexed_images,
        require_primary_coverage,
        resolve_torch_device,
        write_status_error,
    )


def collect_images(directory: Path) -> dict[int, Path]:
    return collect_indexed_images(directory, required_name_token="color")


def load_metric_dependencies():
    try:
        import torch
        import torchvision.transforms.functional as tf
        from lpipsPyTorch import init_lpips_module
        from tqdm import tqdm
        from utils.image_utils import psnr
        from utils.loss_utils import ssim
    except ImportError as exc:
        raise RuntimeError(f"Image metrics dependency is unavailable: {exc}") from exc
    return torch, tf, init_lpips_module, tqdm, psnr, ssim


def image_tensor(path: Path, tf):
    with Image.open(path) as image:
        return tf.to_tensor(image).unsqueeze(0)[:, :3, :, :]


def evaluate(
    render_path: Path,
    gt_path: Path,
    output_path: Path | None = None,
    *,
    device_request: str = "auto",
    allow_partial: bool = False,
) -> bool:
    render_path = Path(render_path)
    gt_path = Path(gt_path)
    out = Path(output_path) if output_path is not None else (render_path / "results.json")
    per_view_path = out.parent / "per_view.json"
    error_path = out.parent / "metric_error.json"
    try:
        render_files = collect_images(render_path)
        gt_files = collect_images(gt_path)
        common_ids = sorted(set(render_files) & set(gt_files))
        missing_render_references, render_coverage = require_primary_coverage(
            render_files,
            common_ids,
            primary_label="render",
            allow_partial=allow_partial,
        )
        missing_gt_renders, gt_coverage = require_primary_coverage(
            gt_files,
            common_ids,
            primary_label="sampled GT",
            allow_partial=allow_partial,
        )
        print(
            f"matched images: {len(common_ids)} "
            f"(render={len(render_files)}, gt={len(gt_files)}, "
            f"render_coverage={render_coverage:.6f}, gt_coverage={gt_coverage:.6f})"
        )
        if not common_ids:
            raise RuntimeError("No matching sampled RGB images")

        torch, tf, init_lpips_module, tqdm, psnr, ssim = load_metric_dependencies()
        device = resolve_torch_device(torch, device_request)
        print(f"metric device: {device}")
        lpips_module = init_lpips_module(device, net_type="vgg").eval()
        ssims: list[float] = []
        psnrs: list[float] = []
        lpipss: list[float] = []
        per_view = {"SSIM": {}, "PSNR": {}, "LPIPS": {}}
        with torch.no_grad():
            for i, idx in enumerate(tqdm(common_ids, desc="Sampled metric evaluation")):
                render_img = image_tensor(render_files[idx], tf).to(device)
                gt_img = image_tensor(gt_files[idx], tf).to(device)
                if render_img.shape != gt_img.shape:
                    raise RuntimeError(
                        f"Image shape mismatch for frame {idx}: "
                        f"render={tuple(render_img.shape)}, gt={tuple(gt_img.shape)}"
                    )
                psnr_value = float(psnr(render_img, gt_img).item())
                ssim_value = float(ssim(render_img, gt_img).item())
                lpips_value = float(lpips_module(2.0 * render_img - 1.0, 2.0 * gt_img - 1.0).item())
                psnrs.append(psnr_value)
                ssims.append(ssim_value)
                lpipss.append(lpips_value)
                image_name = gt_files[idx].name
                per_view["PSNR"][image_name] = psnr_value
                per_view["SSIM"][image_name] = ssim_value
                per_view["LPIPS"][image_name] = lpips_value
                del render_img, gt_img
                if device.type == "cuda" and (i + 1) % 100 == 0:
                    torch.cuda.empty_cache()

        results = {
            "metric_status": "ok",
            "SSIM": float(torch.tensor(ssims).mean().item()),
            "PSNR": float(torch.tensor(psnrs).mean().item()),
            "LPIPS": float(torch.tensor(lpipss).mean().item()),
            "matched_images": len(common_ids),
            "render_images": len(render_files),
            "gt_images": len(gt_files),
            "missing_render_references": missing_render_references,
            "missing_gt_renders": missing_gt_renders,
            "render_coverage": render_coverage,
            "gt_coverage": gt_coverage,
            "device": str(device),
        }
        print("  SSIM : {:>12.7f}".format(results["SSIM"]))
        print("  PSNR : {:>12.7f}".format(results["PSNR"]))
        print("  LPIPS: {:>12.7f}".format(results["LPIPS"]))

        atomic_write_json(out, results)
        atomic_write_json(per_view_path, per_view)
        error_path.unlink(missing_ok=True)
    except Exception as exc:
        out.unlink(missing_ok=True)
        per_view_path.unlink(missing_ok=True)
        if out.parent.is_dir():
            write_status_error(
                error_path,
                exc,
                render_path=str(render_path),
                gt_path=str(gt_path),
            )
        print(f"Unable to compute sampled metrics: {exc}", file=sys.stderr)
        return False
    return True


if __name__ == "__main__":
    parser = ArgumentParser(description="Metrics for sampled eval output folders.")
    parser.add_argument("--render_path", required=True, type=Path)
    parser.add_argument("--gt_path", required=True, type=Path)
    parser.add_argument("--output_path", default=None, type=Path)
    parser.add_argument(
        "--device",
        default="auto",
        help="metric device: auto, cpu, cuda, or cuda:N (default: auto)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="score only the render/GT intersection instead of requiring complete pairing",
    )
    args = parser.parse_args()
    raise SystemExit(
        0
        if evaluate(
            args.render_path,
            args.gt_path,
            args.output_path,
            device_request=args.device,
            allow_partial=args.allow_partial,
        )
        else 1
    )
