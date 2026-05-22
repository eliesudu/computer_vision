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
from basics import histogram_figure_numba


# Example function
# You can use this function to process the images from opencv
# This function must be implemented as a generator function
def custom_processing(img_source_generator):
    # use this figure to plot your histogram
    fig, ax, background, r_plot, g_plot, b_plot = initialize_hist_figure()
    
    for sequence in img_source_generator:
        # Call your custom processing methods here! (e. g. filters)
        
        if keyboard.is_pressed('s'):
            img_to_save = sequence.copy()

            # Falls Bild float mit Werten 0–1 ist
            if img_to_save.dtype != np.uint8:
                if img_to_save.max() <= 1.0:
                    img_to_save = img_to_save * 255

                img_to_save = np.clip(img_to_save, 0, 255).astype(np.uint8)

            filename = f"frame_{int(time.time())}.png"
            path = './VirtualCamera/camera_tests'
            cv2.imwrite(os.path.join(path , filename), img_to_save)
            print("Saved:", filename)

        # Example of keyboard is pressed
        # If you want to use this method then consider implementing a counterwhere pip
        # that ignores for example the next five keyboard press events to
        # "prevent" double clicks due to high fps rates
        if keyboard.is_pressed('h'):
            fig.savefig("histogram.png")
            

        ###
        ### Histogram overlay example (without data)
        ###
        
        # Load the histogram values
        r_bars, g_bars, b_bars = histogram_figure_numba(sequence)        
        
        # Update the histogram with new data
        update_histogram(fig, ax, background, r_plot, g_plot, b_plot, r_bars, g_bars, b_bars)
        
        # uses the figure to create the overlay
        sequence = plot_overlay_to_image(sequence, fig)
        
        ###
        ### END Histogram overlay example
        ###

        
        # Display text example
        display_text_arr = ["Test", "abc"]
        sequence = plot_strings_to_image(sequence, display_text_arr)

        
        # Make sure to yield your processed image
            
        yield sequence



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
            vc.capture_cv_video(0, bgr_to_rgb=True)
            
            # or your window screen
            #vc.capture_screen()
        )
    )

if __name__ == "__main__":
    main()