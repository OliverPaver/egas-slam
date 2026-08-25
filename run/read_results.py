import os
import re
import argparse
import json
import csv


def process_file(file_path):
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        per_frame_time = None
        fps = None
        gpu_memory = None
        local_opt_ms = None
        handoff_opt_ms = None
        gaussian_ops_ms = None

        for line in lines:
            line = line.strip()
            if line.startswith("[PIPELINE AVG TIME]"):
                match = re.search(r"per frame fusion time: (\d+\.\d+)", line)
                if match:
                    per_frame_time = float(match.group(1))
                match = re.search(r"localOptimize time: (\d+\.\d+)", line)
                if match:
                    local_opt_ms = float(match.group(1))
                match = re.search(r"handoffOptimize time: (\d+\.\d+)", line)
                if match:
                    handoff_opt_ms = float(match.group(1))
                match = re.search(r"gaussianOps time: (\d+\.\d+)", line)
                if match:
                    gaussian_ops_ms = float(match.group(1))
                match_fps = re.search(r"FPS: (\d+\.\d+)", line)
                if match_fps:
                    fps = float(match_fps.group(1))
            elif line.startswith("GPU memory usage:"):
                match = re.search(r"GPU memory usage: (\d+) MB", line)
                if match:
                    gpu_memory = int(match.group(1))

        if per_frame_time is None or fps is None or gpu_memory is None:
            print(f"Warning: Skipping {file_path} due to missing data.")
            return None

        calculated_fps = 1000 / per_frame_time
        gaussian_fps = 1000 / (1000 / fps - per_frame_time)
        file_dir = os.path.dirname(file_path)

        ssim = None
        psnr = None
        lpips = None
        json_path = os.path.join(file_dir, "val", "render", "results.json")
        if not os.path.exists(json_path):
            json_path = os.path.join(file_dir, "val", "results.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as jf:
                    json_data = json.load(jf)
                ssim = json_data.get("SSIM")
                psnr = json_data.get("PSNR")
                lpips = json_data.get("LPIPS")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {json_path}: {e}")
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
        else:
            print(f"Note: {json_path} does not exist.")
        res = {
            "RootDir": file_dir,
            "Fusion_FPS": round(calculated_fps, 2),
            "Gaussian_FPS": round(gaussian_fps, 2),
            "FPS": round(fps, 2),
            "LocalOptimize_ms": local_opt_ms,
            "HandoffOptimize_ms": handoff_opt_ms,
            "GaussianOps_ms": gaussian_ops_ms,
            "GPU_Memory": gpu_memory,
            "SSIM": ssim,
            "PSNR": psnr,
            "LPIPS": lpips,
        }
        print(res)
        return res
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Process timelog files")
    parser.add_argument(
        "--root_dir", type=str, help="Root directory to scan for timelog.txt files"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="If provided, only process files with paths containing this string.",
    )
    args = parser.parse_args()

    results = []

    for root, _, files in os.walk(args.root_dir):
        for file in files:
            if file == "time_log.txt":
                file_path = os.path.join(root, file)
                if args.filter and args.filter not in file_path:
                    continue
                data = process_file(file_path)
                if data:
                    results.append(data)

    if results:
        csv_path = os.path.join(args.root_dir, "results.csv")
        fieldnames = [
            "RootDir",
            "Fusion_FPS",
            "Gaussian_FPS",
            "FPS",
            "LocalOptimize_ms",
            "HandoffOptimize_ms",
            "GaussianOps_ms",
            "GPU_Memory",
            "SSIM",
            "PSNR",
            "LPIPS",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {csv_path}")
        means = {key: 0 for key in fieldnames if key != "RootDir"}
        for result in results:
            for key in means:
                means[key] += result[key] if result[key] is not None else 0
        means = {key: value / len(results) for key, value in means.items()}
        print("Means:", means)
    else:
        print("No valid data to save.")


if __name__ == "__main__":
    main()
