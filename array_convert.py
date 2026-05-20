import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

size = 112

def convert_to_array(path, size=size):
    img = cv.imread(path)
    if img is None:
        raise FileNotFoundError(f'Cannot read image at {path}')
    img = cv.resize(img, (size, size))
    return img


def compute_masks(img_bgr):
    """Return tuple of (resized_color, pass_img, failure_img, white_img).

    - pass_img: areas that are green (keeps color where green, black elsewhere)
    - white_img: areas that are white (keeps color where white, black elsewhere)
    - failure_img: areas that are neither green nor white
    """
    # Convert to HSV for color thresholding
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

    # Green mask (tunable)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv.inRange(hsv, lower_green, upper_green)

    # White mask (low saturation, high value)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    white_mask = cv.inRange(hsv, lower_white, upper_white)

    # Failure = not green and not white
    combined = cv.bitwise_or(green_mask, white_mask)
    failure_mask = cv.bitwise_not(combined)

    pass_img = cv.bitwise_and(img_bgr, img_bgr, mask=green_mask)
    white_img = cv.bitwise_and(img_bgr, img_bgr, mask=white_mask)

    # Render failure areas as solid red for clear visibility
    failure_img = np.zeros_like(img_bgr)
    failure_img[failure_mask > 0] = (0, 0, 255)  # BGR red

    return img_bgr, pass_img, failure_img, white_img


def plot_four(resized, pass_img, failure_img, white_img, figsize=(10,8)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()

    images = [resized, pass_img, failure_img, white_img]
    titles = ['Resized', 'Pass', 'Failure', 'White area']

    for ax, img, t in zip(axes, images, titles):
        # convert BGR -> RGB for display
        if img is None:
            ax.set_visible(False)
            continue
        disp = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        ax.imshow(disp)
        ax.set_title(t)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compute and display wafer masks using matplotlib')
    parser.add_argument('--input', '-i', default='wafer.png', help='Path to wafer image')
    parser.add_argument('--size', type=int, default=size, help='Resize dimension for processing')
    parser.add_argument('--show-original', action='store_true', help='Also show the original image in a single figure')
    args = parser.parse_args()

    path = args.input
    img_array = convert_to_array(path, size=args.size)
    print('Image shape (H,W,C):', img_array.shape)

    # compute masks and show 4-panel plot
    resized, pass_img, failure_img, white_img = compute_masks(img_array)
    plot_four(resized, pass_img, failure_img, white_img)

    # optionally show the original image alone (matplotlib expects RGB)
    if args.show_original:
        rgb = cv.cvtColor(img_array, cv.COLOR_BGR2RGB)
        plt.figure(figsize=(6,6))
        plt.imshow(rgb)
        plt.title(f'Original: {path}')
        plt.axis('off')
        plt.show()
