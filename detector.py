import json
import os.path
import sys
import typing

import cv2
import numpy as np

Point = typing.Tuple[int, int]
Rect = typing.Tuple[Point, Point]
Config = typing.Dict[str, typing.Any]


def gradient(img: np.ndarray) -> np.ndarray:
    hor_grad = (img[1:, :] - img[:-1, :])[:, :-1]
    ver_grad = (img[:, 1:] - img[:, :-1])[:-1:, :]
    magnitude = np.sqrt(
        np.linalg.norm(hor_grad, axis=-1) ** 2
        + np.linalg.norm(ver_grad, axis=-1) ** 2,
    )
    return magnitude


INT_SUM_MAX = 10 ** 18


def integral_sum(img: np.ndarray) -> np.ndarray:
    sum_img = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            top_sum = sum_img[i, j - 1] if j > 0 else 0
            left_sum = sum_img[i - 1, j] if i > 0 else 0
            top_left_sum = sum_img[i - 1, j - 1] if i > 0 and j > 0 else 0
            sum_img[i, j] = (
                left_sum + (top_sum - top_left_sum) + img[i, j]
            ) % INT_SUM_MAX
    return sum_img


def get_sum(rect: Rect, sum_img: np.ndarray):
    top_sum = sum_img[rect[1][0], rect[0][1]] if rect[0][1] > 0 else 0
    left_sum = sum_img[rect[0][0], rect[1][1]] if rect[0][0] > 0 else 0
    top_left_sum = (
        sum_img[rect[0][0], rect[0][1]]
        if rect[0][0] > 0 and rect[0][1] > 0
        else 0
    )
    return np.sum(
        (
            (sum_img[rect[1][0], rect[1][1]] - top_sum)
            + (top_left_sum - left_sum)
        )
        % INT_SUM_MAX,
    )


def detect_color_gradient(
        sum_img: np.ndarray, gradient_img: np.ndarray, config: Config,
) -> typing.List[Rect]:
    rects: typing.List[Rect] = []
    max_area = 0
    nearest_top = -(np.ones(gradient_img.shape[1]).astype(int))
    nearest_left = -(np.ones(gradient_img.shape[1]).astype(int))
    nearest_right = (
        np.ones(gradient_img.shape[1]).astype(int) * gradient_img.shape[1]
    )
    for i in range(gradient_img.shape[0]):
        # Calc nearest_top
        for j in range(gradient_img.shape[1]):
            if gradient_img[i][j] > config['grad_max']:
                nearest_top[j] = i

        # Calc nearest_left
        stack = []
        for j in range(gradient_img.shape[1]):
            while stack and nearest_top[stack[-1]] <= nearest_top[j]:
                stack.pop()
            if stack:
                nearest_left[j] = stack[-1]
            stack.append(j)

        # Calc nearest_right
        stack = []
        for j in reversed(range(gradient_img.shape[1])):
            while stack and nearest_top[stack[-1]] <= nearest_top[j]:
                stack.pop()
            if stack:
                nearest_right[j] = stack[-1]
            stack.append(j)

        # Find suitable rect with max area
        for j in range(gradient_img.shape[1]):
            rect = (
                (nearest_top[j] + 1, nearest_left[j] + 1),
                (i, nearest_right[j] - 1),
            )
            area = (rect[1][0] - rect[0][0]) * (rect[1][1] - rect[0][1])
            if area > 0:
                rect_avg = get_sum(rect, sum_img) / area
                if area > max_area and rect_avg > config['min_avg']:
                    max_area = area
                    rects = [rect]

    return rects


def main():
    assert len(sys.argv) == 2
    src_path = sys.argv[1]
    assert os.path.exists(src_path)
    img = cv2.imread(src_path)
    assert img is not None
    img = img.astype(np.float64)

    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    gradient_img = gradient(img)
    if 'gradient_dir' in config:
        out_dir = config['gradient_dir']
        assert os.path.exists(out_dir)
        dst_path = os.path.join(out_dir, os.path.split(src_path)[-1])
        cv2.imwrite(dst_path, gradient_img)

    sum_img = integral_sum(img.astype(np.float64))

    rects = detect_color_gradient(sum_img, gradient_img, config)
    print(rects)
    if 'rect_dir' in config:
        for rect in rects:
            img = cv2.rectangle(
                img,
                (rect[0][1], rect[0][0]),
                (rect[1][1], rect[1][0]),
                color=config['rect_color'],
                thickness=config['rect_thickness'],
            )
        out_dir = config['rect_dir']
        assert os.path.exists(out_dir)
        dst_path = os.path.join(out_dir, os.path.split(src_path)[-1])
        cv2.imwrite(dst_path, img)


if __name__ == '__main__':
    main()
