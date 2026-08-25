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


def matched_image_paths(renders_dir, gt_dir, allow_partial=False):
    rgb_dir = Path(gt_dir) / "camera"
    depth_dir = Path(gt_dir) / "depth"
    renders_dir = Path(renders_dir)
    gt_files = collect_indexed_images(rgb_dir)
    depth_files = collect_indexed_images(depth_dir)
    render_files = collect_indexed_images(renders_dir)

    common_ids = sorted(set(render_files) & set(gt_files) & set(depth_files))
    missing_render_references, render_coverage = require_primary_coverage(
        render_files,
        common_ids,
        primary_label="render",
        allow_partial=allow_partial,
    )
    print(
        f"matched images: {len(common_ids)} "
        f"(render={len(render_files)}, gt={len(gt_files)}, depth={len(depth_files)}, "
        f"render_coverage={render_coverage:.6f})"
    )
    if not common_ids:
        raise RuntimeError("No matching render, RGB, and depth images")
    paths = [(idx, render_files[idx], gt_files[idx], depth_files[idx]) for idx in common_ids]
    counts = {
        "matched_images": len(common_ids),
        "render_images": len(render_files),
        "gt_images": len(gt_files),
        "depth_images": len(depth_files),
        "missing_render_references": missing_render_references,
        "render_coverage": render_coverage,
    }
    return paths, counts


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


def depth_tensor(path, tf):
    with Image.open(path) as image:
        return tf.to_tensor(image).unsqueeze(0)


def evaluate(renders_dir, gt_dir, depth_mask=True, device_request="auto", allow_partial=False):
    renders_dir = Path(renders_dir)
    results_path = renders_dir / "results.json"
    per_view_path = renders_dir / "per_view.json"
    error_path = renders_dir / "metric_error.json"
    if depth_mask:
        print("ignore invalid depth!!!")

    try:
        print("Processing images from:", renders_dir, gt_dir)
        ssims = []
        psnrs = []
        lpipss = []
        per_view_ssim = {}
        per_view_psnr = {}
        per_view_lpips = {}
        image_paths, coverage = matched_image_paths(
            renders_dir,
            Path(gt_dir),
            allow_partial=allow_partial,
        )
        torch, tf, init_lpips_module, tqdm, psnr, ssim = load_metric_dependencies()
        device = resolve_torch_device(torch, device_request)
        print(f"metric device: {device}")
        lpips_module = init_lpips_module(device, net_type="vgg").eval()
        with torch.no_grad():
            for idx, (_frame_idx, render_path, gt_path, depth_path) in enumerate(tqdm(image_paths, desc="Metric evaluation progress")):
                render_img = image_tensor(render_path, tf).to(device)
                gt_img = image_tensor(gt_path, tf).to(device)
                depth_img = depth_tensor(depth_path, tf).to(device)
                if render_img.shape != gt_img.shape:
                    raise RuntimeError(
                        f"Image shape mismatch for frame {_frame_idx}: "
                        f"render={tuple(render_img.shape)}, gt={tuple(gt_img.shape)}"
                    )
                if depth_img.shape[-2:] != gt_img.shape[-2:]:
                    raise RuntimeError(
                        f"Depth shape mismatch for frame {_frame_idx}: "
                        f"depth={tuple(depth_img.shape)}, gt={tuple(gt_img.shape)}"
                    )
                if depth_mask:
                    render_img = render_img * (depth_img > 0)
                    gt_img = gt_img * (depth_img > 0)
                psnr_value = psnr(render_img, gt_img).item()
                ssim_value = ssim(render_img, gt_img).item()
                lpips_value = lpips_module(render_img, gt_img).item()
                image_name = gt_path.name
                psnrs.append(psnr_value)
                ssims.append(ssim_value)
                lpipss.append(lpips_value)
                per_view_psnr[image_name] = psnr_value
                per_view_ssim[image_name] = ssim_value
                per_view_lpips[image_name] = lpips_value
                del render_img, gt_img, depth_img
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
            "depth_mask": bool(depth_mask),
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
        if renders_dir.is_dir():
            write_status_error(
                error_path,
                e,
                render_path=str(renders_dir),
                gt_path=str(gt_dir),
            )
        print(
            f"Unable to compute metrics for render path {renders_dir}. Error: {str(e)}",
            file=sys.stderr,
        )
        return False
    return True


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Evaluation script parameters")
    parser.add_argument(
        "--gt_path",
        required=True,
        type=str,
        help="Path to the directory containing gt images",
    )
    parser.add_argument(
        "--render_path",
        required=True,
        type=str,
        help="Path to the directory containing render images",
    )
    parser.add_argument(
        "--depth_mask",
        action="store_true",
        help="Use depth mask if provided",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="metric device: auto, cpu, cuda, or cuda:N (default: auto)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="score the intersection even when one or more render frames lack GT/depth",
    )
    args = parser.parse_args()

    # Pass the depth_mask argument to the evaluate function
    raise SystemExit(
        0
        if evaluate(
            args.render_path,
            args.gt_path,
            depth_mask=args.depth_mask,
            device_request=args.device,
            allow_partial=args.allow_partial,
        )
        else 1
    )
