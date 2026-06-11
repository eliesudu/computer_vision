# -*- coding: utf-8 -*-
"""
Created on Mon May  3 19:18:29 2021

@author: droes
"""
from numba import njit # conda install numba
import numpy as np
import cv2
from matplotlib import pyplot as plt


def image_statistics(np_img):

    img = np_img[:, :, :3].astype(np.uint8)

    stats = {}
    channel_names = ["R", "G", "B"]

    for i, name in enumerate(channel_names):
        values = img[:, :, i].ravel()

        counts = np.bincount(values, minlength=256)
        mode = np.argmax(counts)

        stats[name] = {
            "mean": float(np.mean(values)),
            "mode": int(mode),
            "std": float(np.std(values)),
            "min": int(np.min(values)),
            "max": int(np.max(values))
        }

    return stats


def linear_transformation(np_img, alpha=1.2, beta=10):

    img = np_img.astype(np.float32)
    transformed = alpha * img + beta
    transformed = np.clip(transformed, 0, 255)
    return transformed.astype(np.uint8)


def entropy(np_img):

    img = np_img[:, :, :3].astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probabilities = hist / np.sum(hist)

    probabilities = probabilities[probabilities > 0]

    return float(-np.sum(probabilities * np.log2(probabilities)))


@njit
def histogram_figure_numba(np_img):

    h, w, c = np_img.shape

    r_hist = np.zeros(256)
    g_hist = np.zeros(256)
    b_hist = np.zeros(256)

    for y in range(h):
        for x in range(w):
            r = int(np_img[y, x, 0])
            g = int(np_img[y, x, 1])
            b = int(np_img[y, x, 2])

            r_hist[r] += 1
            g_hist[g] += 1
            b_hist[b] += 1

    max_r = np.max(r_hist)
    max_g = np.max(g_hist)
    max_b = np.max(b_hist)

    if max_r > 0:
        r_hist = r_hist / max_r * 3.0

    if max_g > 0:
        g_hist = g_hist / max_g * 3.0

    if max_b > 0:
        b_hist = b_hist / max_b * 3.0

    return r_hist, g_hist, b_hist


def equalize_histogram(np_img):
    img = np_img[:, :, :3].astype(np.uint8)

    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])

    equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return equalized


def blur_filter(np_img, kernel_size=5):
    # gauss blur
    return cv2.GaussianBlur(np_img, (kernel_size, kernel_size), 0)


def sharpen_filter(np_img):

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    return cv2.filter2D(np_img, -1, kernel)


def sobel_filter(np_img):

    img = np_img[:, :, :3].astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

    return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2RGB)

def save_rgb_histogram(np_img, filename="histogram.png"):
    img = np_img[:, :, :3].astype(np.uint8)

    plt.figure(figsize=(8, 5))

    channel_names = ["R", "G", "B"]
    colors = ["red", "green", "blue"]

    for i in range(3):
        values = img[:, :, i].ravel()
        hist = np.bincount(values, minlength=256)

        plt.plot(hist, color=colors[i], label=channel_names[i])

    plt.title("RGB Histogram")
    plt.xlabel("Pixel intensity")
    plt.ylabel("Frequency")
    plt.xlim([0, 255])
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()