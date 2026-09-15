from pathlib import Path
import sys
from PIL import Image
from argparse import ArgumentParser

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


def matched_image_paths(input_path, allow_partial=False):
    # renders_dir = input_path / "raycast_color"
    renders_dir = input_path / "render"
    gt_dir = input_path / "gt"
    if not renders_dir.exists() or not gt_dir.exists():
        raise RuntimeError(f"Missing render or gt directory under {input_path}")
    render_files = collect_indexed_images(renders_dir, required_name_token="color")
    gt_files = collect_indexed_images(gt_dir, required_name_token="color")
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
        primary_label="GT",
        allow_partial=allow_partial,
    )
    image_paths = [
        (idx, gt_files[idx].name, render_files[idx], gt_files[idx])
        for idx in common_ids
    ]
    if not image_paths:
        raise RuntimeError(f"No matching render and GT color images under {input_path}")
    print(
        f"matched images: {len(image_paths)} "
        f"(render={len(render_files)}, gt={len(gt_files)}, "
        f"render_coverage={render_coverage:.6f}, gt_coverage={gt_coverage:.6f})"
    )
    return image_paths, {
        "matched_images": len(image_paths),
        "render_images": len(render_files),
        "gt_images": len(gt_files),
        "missing_render_references": missing_render_references,
        "missing_gt_renders": missing_gt_renders,
        "render_coverage": render_coverage,
        "gt_coverage": gt_coverage,
    }


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


def image_tensor(path, tf):
    with Image.open(path) as image:
        return tf.to_tensor(image).unsqueeze(0)[:, :3, :, :]


def evaluate(input_path, device_request="auto", allow_partial=False):
    input_path = Path(input_path)
    results_path = input_path / "results.json"
    per_view_path = input_path / "per_view.json"
    error_path = input_path / "metric_error.json"

    try:
        print("Processing images from:", input_path)
        ssims = []
        psnrs = []
        lpipss = []
        per_view_ssim = {}
        per_view_psnr = {}
        per_view_lpips = {}
        image_paths, coverage = matched_image_paths(input_path, allow_partial=allow_partial)
        torch, tf, init_lpips_module, tqdm, psnr, ssim = load_metric_dependencies()
        device = resolve_torch_device(torch, device_request)
        print(f"metric device: {device}")
        lpips_module = init_lpips_module(device, net_type="vgg").eval()
        with torch.no_grad():
            for idx, (frame_idx, image_name, render_path, gt_path) in enumerate(tqdm(image_paths, desc="Metric evaluation progress")):
                render_img = image_tensor(render_path, tf).to(device)
                gt_img = image_tensor(gt_path, tf).to(device)
                if render_img.shape != gt_img.shape:
                    raise RuntimeError(
                        f"Image shape mismatch for frame {frame_idx}: "
                        f"render={tuple(render_img.shape)}, gt={tuple(gt_img.shape)}"
                    )
                psnr_value = psnr(render_img, gt_img).item()
                ssim_value = ssim(render_img, gt_img).item()
                lpips_value = lpips_module(2.0 * render_img - 1.0, 2.0 * gt_img - 1.0).item()
                psnrs.append(psnr_value)
                ssims.append(ssim_value)
                lpipss.append(lpips_value)
                per_view_psnr[image_name] = psnr_value
                per_view_ssim[image_name] = ssim_value
                per_view_lpips[image_name] = lpips_value
                del render_img, gt_img
                if device.type == "cuda" and (idx + 1) % 100 == 0:
                    torch.cuda.empty_cache()

        mean_ssim = float(torch.tensor(ssims).mean().item())
        mean_psnr = float(torch.tensor(psnrs).mean().item())
        mean_lpips = float(torch.tensor(lpipss).mean().item())
        print("  SSIM : {:>12.7f}".format(mean_ssim, ".5"))
        print("  PSNR : {:>12.7f}".format(mean_psnr, ".5"))
        print("  LPIPS: {:>12.7f}".format(mean_lpips, ".5"))
        print("")

        full_dict = {
            "metric_status": "ok",
            "SSIM": mean_ssim,
            "PSNR": mean_psnr,
            "LPIPS": mean_lpips,
            "device": str(device),
            **coverage,
        }
        per_view_dict = {
            "SSIM": per_view_ssim,
            "PSNR": per_view_psnr,
            "LPIPS": per_view_lpips,
        }

        atomic_write_json(results_path, full_dict)
        atomic_write_json(per_view_path, per_view_dict)
        error_path.unlink(missing_ok=True)
    except Exception as e:
        results_path.unlink(missing_ok=True)
        per_view_path.unlink(missing_ok=True)
        if input_path.is_dir():
            write_status_error(error_path, e, input_path=str(input_path))
        print(
            f"Unable to compute metrics for input path {input_path}. Error: {str(e)}",
            file=sys.stderr,
        )
        return False
    return True


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Evaluation script parameters")
    parser.add_argument(
        "--input_path",
        "-i",
        required=True,
        type=str,
        help="Path to the directory containing 'gt' and 'renders' subdirectories",
    )
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
            args.input_path,
            device_request=args.device,
            allow_partial=args.allow_partial,
        )
        else 1
    )
