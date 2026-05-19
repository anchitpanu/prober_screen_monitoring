"""
Wafer scan simulator.

Reveals random pixels from an input image in random step sizes
to simulate progressive scan acquisition.

Usage:
	python Scan_Sim.py --input wafer.png --size 112 --min-step 0.01 --max-step 0.04 --display-scale 4
"""

import argparse
import os
import random
import cv2 as cv
import numpy as np
from array_convert import convert_to_array, compute_masks


def create_writer(video_path, width, height, fps):
	if not video_path:
		return None
	fourcc = cv.VideoWriter_fourcc(*"mp4v")
	writer = cv.VideoWriter(video_path, fourcc, fps, (width, height))
	if not writer.isOpened():
		raise RuntimeError(f"Cannot create video file: {video_path}")
	return writer


def run_scan_simulation(image, min_step_pct, max_step_pct, delay_ms=60, seed=None, video_path=None, display_scale=4, cluster_max=3):
	if seed is not None:
		random.seed(seed)
		np.random.seed(seed)

	h, w = image.shape[:2]
	total_pixels = h * w
	_, _, _, white_img = compute_masks(image)
	white_mask = np.any(white_img > 0, axis=2)

	# Hidden canvas starts as white image (unrevealed area shown as white).
	# Keep white-mask areas visually white (they are considered pre-revealed
	# but we do not copy underlying image pixels here so they match unrevealed).
	canvas = np.full_like(image, 255)

	# Track unrevealed pixel positions in flattened index space.
	white_flat = np.flatnonzero(white_mask.ravel())
	all_flat = np.arange(total_pixels, dtype=np.int32)
	revealed_mask = np.zeros((h, w), dtype=bool)
	if white_flat.size:
		revealed_mask.flat[white_flat] = True
	revealed_count = int(revealed_mask.sum())
	unrevealed = np.flatnonzero(~revealed_mask.ravel())
	step_idx = 0

	display_scale = max(1, int(display_scale))
	disp_w = w * display_scale
	disp_h = h * display_scale

	fps = max(1, int(round(1000 / max(1, delay_ms))))
	writer = create_writer(video_path, disp_w, disp_h, fps)

	win = "Wafer Scan Simulator"
	cv.namedWindow(win, cv.WINDOW_AUTOSIZE)

	try:
		# Initial frame (white area already revealed). Wait for SPACE to start.
		start_frame = cv.resize(canvas, (disp_w, disp_h), interpolation=cv.INTER_NEAREST)
		cv.imshow(win, start_frame)
		if writer is not None:
			writer.write(start_frame)

		while unrevealed.size > 0:
			key = cv.waitKey(0) & 0xFF
			if key == ord("q"):
				break
			if key != ord(" "):
				continue

			step_idx += 1

			step_pct = random.uniform(min_step_pct, max_step_pct)
			step_pixels = max(1, int(round((step_pct / 100.0) * total_pixels)))
			step_pixels = min(step_pixels, unrevealed.size)

			remaining = step_pixels
			# reveal clusters until we satisfy step_pixels or run out
			while remaining > 0 and unrevealed.size > 0:
				# choose a random unrevealed pixel as cluster center
				center_flat = int(np.random.choice(unrevealed))
				cy = center_flat // w
				sx = center_flat % w
				# random radius between 1 and cluster_max
				r = random.randint(1, max(1, int(cluster_max)))
				# bounding box
				y0 = max(0, cy - r)
				y1 = min(h - 1, cy + r)
				x0 = max(0, sx - r)
				x1 = min(w - 1, sx + r)
				# compute disk mask within bbox
				ys = np.arange(y0, y1 + 1)
				xs = np.arange(x0, x1 + 1)
				yy, xx = np.meshgrid(ys, xs, indexing='ij')
				dists = np.sqrt((yy - cy) ** 2 + (xx - sx) ** 2)
				mask = dists <= r
				# get flat indices for candidate pixels
				cand_flats = (yy * w + xx).ravel()
				cand_mask = mask.ravel()
				cand_flats = cand_flats[cand_mask]
				# filter only currently unrevealed
				is_unrevealed = ~revealed_mask.flat[cand_flats]
				cand_flats = cand_flats[is_unrevealed]
				if cand_flats.size == 0:
					# nothing new in this cluster; try different center
					unrevealed = np.flatnonzero(~revealed_mask.ravel())
					continue
				# how many to take from this cluster
				take = min(remaining, cand_flats.size)
				pick_idxs = np.random.choice(cand_flats.size, size=take, replace=False)
				pick_flats = cand_flats[pick_idxs]
				# reveal them
				ys2 = pick_flats // w
				xs2 = pick_flats % w
				canvas[ys2, xs2] = image[ys2, xs2]
				revealed_mask.flat[pick_flats] = True
				revealed_count += pick_flats.size
				remaining -= pick_flats.size
				# update unrevealed list
				unrevealed = np.flatnonzero(~revealed_mask.ravel())
			# (clusters already revealed in the loop above)
			frame = cv.resize(canvas, (disp_w, disp_h), interpolation=cv.INTER_NEAREST)

			cv.imshow(win, frame)
			if writer is not None:
				writer.write(frame)

		# Show final frame at completion unless user quit early.
		if unrevealed.size == 0:
			final = cv.resize(canvas, (disp_w, disp_h), interpolation=cv.INTER_NEAREST)
			cv.imshow(win, final)
			if writer is not None:
				writer.write(final)
			cv.waitKey(0)
	finally:
		if writer is not None:
			writer.release()
		cv.destroyAllWindows()


def parse_args():
	parser = argparse.ArgumentParser(description="Simulate wafer scan by random pixel reveal")
	parser.add_argument("--input", "-i", required=True, help="Input image path")
	parser.add_argument("--size", type=int, default=112, help="Square resize used by array conversion (default: 112)")
	parser.add_argument("--min-step", type=float, default=0.2, help="Minimum reveal percentage per step")
	parser.add_argument("--max-step", type=float, default=1.0, help="Maximum reveal percentage per step")
	parser.add_argument("--delay", type=int, default=60, help="Used only to set video FPS when --video is enabled")
	parser.add_argument("--display-scale", type=int, default=4, help="Upscale factor for display (default: 4)")
	parser.add_argument("--seed", type=int, default=None, help="Random seed for repeatable simulation")
	parser.add_argument("--video", type=str, default=None, help="Optional output MP4 file path")
	return parser.parse_args()


def main():
	args = parse_args()

	if args.min_step <= 0 or args.max_step <= 0:
		raise ValueError("--min-step and --max-step must be > 0")
	if args.min_step > args.max_step:
		raise ValueError("--min-step cannot be greater than --max-step")
	if args.delay < 1:
		raise ValueError("--delay must be >= 1 ms")
	if args.size < 1:
		raise ValueError("--size must be >= 1")
	if args.display_scale < 1:
		raise ValueError("--display-scale must be >= 1")

	if not os.path.isfile(args.input):
		raise FileNotFoundError(f"Input image not found: {args.input}")

	# Convert image to array exactly like array_convert.py before simulation.
	image = convert_to_array(args.input, size=args.size)

	run_scan_simulation(
		image=image,
		min_step_pct=args.min_step,
		max_step_pct=args.max_step,
		delay_ms=args.delay,
		seed=args.seed,
		video_path=args.video,
		display_scale=args.display_scale,
	)


if __name__ == "__main__":
	main()
