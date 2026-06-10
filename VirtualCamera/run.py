# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:59:19 2021

@author: droes
"""
# You can use this library for oberserving keyboard presses
import keyboard # pip install hkeyboard
import cv2
import numpy as np
import os
import time
from capturing import VirtualCamera
from overlays import initialize_hist_figure, plot_overlay_to_image, plot_strings_to_image, update_histogram

from basics import (
    histogram_figure_numba,
    image_statistics,
    linear_transformation,
    entropy,
    equalize_histogram,
    sharpen_filter,
    sobel_filter,
    blur_filter,
    save_rgb_histogram
)


def screenshot(sequence, key):
    if keyboard.is_pressed(key):
            img_to_save = sequence.copy()

            # Falls Bild float mit Werten 0–1 ist
            if img_to_save.dtype != np.uint8:
                if img_to_save.max() <= 1.0:
                    img_to_save = img_to_save * 255

                img_to_save = np.clip(img_to_save, 0, 255).astype(np.uint8)
            

            
            filename = f"frame_{int(time.time())}.png"
            path = './VirtualCamera/camera_tests'

            img_to_save_bgr = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(path, filename), img_to_save_bgr)
            # cv2.imwrite(os.path.join(path , filename), img_to_save) RGB
            print("Saved:", filename)


# Example function
# You can use this function to process the images from opencv
# This function must be implemented as a generator function
def custom_processing(img_source_generator):
    # use this figure to plot your histogram
    fig, ax, background, r_plot, g_plot, b_plot = initialize_hist_figure()

    for sequence in img_source_generator:
        screenshot(sequence, key="s")

        img = sequence.copy()

        stats = image_statistics(img)
        ent = entropy(img)

        #img = linear_transformation(img, alpha=1.1, beta=5)

        img = equalize_histogram(img)

        #img = sharpen_filter(img)

        r_bars, g_bars, b_bars = histogram_figure_numba(img)

        update_histogram(
            fig,
            ax,
            background,
            r_plot,
            g_plot,
            b_plot,
            r_bars,
            g_bars,
            b_bars
        )

        img = plot_overlay_to_image(img, fig)

        display_text_arr = [
            f"Entropy: {ent:.2f}",

            f"R mean: {stats['R']['mean']:.1f}, std: {stats['R']['std']:.1f}",
            f"G mean: {stats['G']['mean']:.1f}, std: {stats['G']['std']:.1f}",
            f"B mean: {stats['B']['mean']:.1f}, std: {stats['B']['std']:.1f}",

            f"R mode: {stats['R']['mode']}, min/max: {stats['R']['min']}/{stats['R']['max']}",
            f"G mode: {stats['G']['mode']}, min/max: {stats['G']['min']}/{stats['G']['max']}",
            f"B mode: {stats['B']['mode']}, min/max: {stats['B']['min']}/{stats['B']['max']}",
        ]

        img = plot_strings_to_image(img, display_text_arr)


        cv2.imshow("Processed Frame", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        if keyboard.is_pressed('h'):
            save_rgb_histogram(img, "histogram.png")
            
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return
        
        yield img



def main():
    # change according to your settings
    width = 1280
    height = 720
    fps = 30
    
    # Define your virtual camera
    vc = VirtualCamera(fps, width, height)
    
    vc.virtual_cam_interaction(
        custom_processing(
            # either camera stream
            vc.capture_cv_video(bgr_to_rgb=True)
            
            # or your window screen
            #vc.capture_screen()
        )
    )

if __name__ == "__main__":
    main()