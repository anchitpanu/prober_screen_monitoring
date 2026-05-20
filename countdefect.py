import argparse
import cv2
import numpy as np
import os


def compute_image_metrics(image):
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    green_pixel_count = cv2.countNonZero(green_mask)
    red_pixel_count = cv2.countNonZero(red_mask)
    total_pixels = image.shape[0] * image.shape[1]

    green_density = (green_pixel_count / total_pixels) * 100
    red_density = (red_pixel_count / total_pixels) * 100
    ratio = red_pixel_count / green_pixel_count if green_pixel_count > 0 else float('inf')

    return green_density, red_density, ratio


def classify_metrics(green_density, red_density, ratio, strategy, green_threshold, red_threshold, ratio_threshold):
    if strategy == "green":
        return "pass" if green_density >= green_threshold else "fail"
    if strategy == "red":
        return "fail" if red_density >= red_threshold else "pass"
    if strategy == "ratio":
        return "fail" if ratio >= ratio_threshold else "pass"
    if strategy == "combined":
        if green_density >= green_threshold and red_density <= red_threshold and ratio <= ratio_threshold:
            return "pass"
        return "fail"
    raise ValueError(f"Unknown strategy: {strategy}")


def collect_metrics(folder_path, label):
    results = []
    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith('.png'):
            continue

        file_path = os.path.join(folder_path, file_name)
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Could not open image: {file_path}")

        green_density, red_density, ratio = compute_image_metrics(image)
        results.append({
            "filename": file_name,
            "true_label": label,
            "green_density": green_density,
            "red_density": red_density,
            "ratio": ratio,
        })
    return results
    


def evaluate_thresholds(pass_folder, fail_folder, strategy, green_threshold, red_threshold, ratio_threshold):
    pass_results = collect_metrics(pass_folder, "pass")
    fail_results = collect_metrics(fail_folder, "fail")
    all_results = pass_results + fail_results

    summary = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "total": len(all_results)}

    for result in all_results:
        predicted = classify_metrics(
            result["green_density"],
            result["red_density"],
            result["ratio"],
            strategy,
            green_threshold,
            red_threshold,
            ratio_threshold,
        )
        result["predicted"] = predicted

        if result["true_label"] == "pass" and predicted == "pass":
            summary["TP"] += 1
        elif result["true_label"] == "pass" and predicted == "fail":
            summary["FN"] += 1
        elif result["true_label"] == "fail" and predicted == "fail":
            summary["TN"] += 1
        elif result["true_label"] == "fail" and predicted == "pass":
            summary["FP"] += 1

    summary["accuracy"] = (summary["TP"] + summary["TN"]) / summary["total"] if summary["total"] else 0.0
    summary["precision"] = summary["TP"] / (summary["TP"] + summary["FP"]) if (summary["TP"] + summary["FP"]) else 0.0
    summary["recall"] = summary["TP"] / (summary["TP"] + summary["FN"]) if (summary["TP"] + summary["FN"]) else 0.0
    summary["f1"] = (2 * summary["precision"] * summary["recall"]) / (summary["precision"] + summary["recall"]) if (summary["precision"] + summary["recall"]) else 0.0

    return all_results, summary


def print_evaluation(all_results, summary, strategy, green_threshold, red_threshold, ratio_threshold):
    print("Threshold hypothesis results")
    print("----------------------------")
    print(f"Strategy: {strategy}")
    print(f"Green threshold: {green_threshold:.4f}%")
    print(f"Red threshold:   {red_threshold:.4f}%")
    print(f"Ratio threshold: {ratio_threshold:.4f}")
    print()
    print(f"Total images: {summary['total']}")
    print(f"True positives (pass predicted pass): {summary['TP']}")
    print(f"True negatives (fail predicted fail): {summary['TN']}")
    print(f"False positives (fail predicted pass): {summary['FP']}")
    print(f"False negatives (pass predicted fail): {summary['FN']}")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Precision: {summary['precision']:.4f}")
    print(f"Recall: {summary['recall']:.4f}")
    print(f"F1 score: {summary['f1']:.4f}")
    print()

    if summary['total']:
        print("Sample predictions:")
        for result in all_results[:5]:
            print(
                f"{result['filename']}: true={result['true_label']} "
                f"pred={result['predicted']} "
                f"green={result['green_density']:.2f}% "
                f"red={result['red_density']:.2f}% "
                f"ratio={result['ratio']:.2f}"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pass/fail thresholds for wafer images")
    parser.add_argument("--pass-dir", default="classified/incompleted_test/pass", help="Folder containing pass images")
    parser.add_argument("--fail-dir", default="classified/incompleted_test/fail", help="Folder containing fail images")
    parser.add_argument("--strategy", choices=["green", "red", "ratio", "combined"], default="combined", help="Threshold strategy to evaluate")
    parser.add_argument("--green-threshold", type=float, default=45.0, help="Minimum green density to consider pass")
    parser.add_argument("--red-threshold", type=float, default=3.0, help="Maximum red density to consider pass")
    parser.add_argument("--ratio-threshold", type=float, default=0.05, help="Maximum red/green ratio to consider pass")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.pass_dir):
        raise FileNotFoundError(f"Pass folder not found: {args.pass_dir}")
    if not os.path.isdir(args.fail_dir):
        raise FileNotFoundError(f"Fail folder not found: {args.fail_dir}")

    all_results, summary = evaluate_thresholds(
        args.pass_dir,
        args.fail_dir,
        args.strategy,
        args.green_threshold,
        args.red_threshold,
        args.ratio_threshold,
    )
    print_evaluation(all_results, summary, args.strategy, args.green_threshold, args.red_threshold, args.ratio_threshold)


if __name__ == '__main__':
    main()
