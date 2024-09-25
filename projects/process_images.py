# Main file for processing image input from simulation and camera output
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import tkinter as tk
import seaborn as sns
from core.object_locations import *

from improc.improc import (read_images, simulate_mosaicing_rggb, get_colour_histograms,
                           white_balance_adjust, auto_colour_adjust, process_images, reshape_img)
from improc.ssim_cropper import ImageCropper, all_rects
from core.object_locations import *

sns.set(palette="dark", font_scale=1.1, color_codes=True)
sns.set_style('darkgrid', {'axes.linewidth': 1, 'axes.edgecolor': 'black'})

matplotlib.use('TkAgg')

matplotlib.use('TkAgg')


def visualise_ssim_regions(img1, img2, view=None, area=None):
    """Tool for visualising processed SSIM regions"""
    if area is None:
        area = all_rects
    elif area == 'full':
        area = [[0, 0, img1.shape[1], img1.shape[0]], [0, 0, img2.shape[1], img2.shape[0]]]

    # Firstly reshape images
    img2 = reshape_img(img1, img2)

    # BGR2RGB
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Perform simulated mosaicing
    img1_mos = simulate_mosaicing_rggb(img1, blend=0.75)

    # Greyscale
    img1_g = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_g = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Colour correction
    img1_arr = np.array(img1)
    img2_arr = np.array(img2)
    img1_arr = white_balance_adjust(img1_arr, img2)
    img1_arr = auto_colour_adjust(img1_arr, img2_arr, blend=0.0)

    # Perform simulated mosaicing
    img1_arr = simulate_mosaicing_rggb(img1_arr, blend=0.75)

    # plt.figure(figsize=(10, 6))
    # plt.title('Simulated image with colour correction')
    # plt.grid(False)
    # plt.imshow(img1_arr)
    # plt.show()

    # Convert back to greyscale for comparison
    img1_g_2 = cv2.cvtColor(img1_arr, cv2.COLOR_RGB2GRAY)
    img2_g_2 = cv2.cvtColor(img2_arr, cv2.COLOR_RGB2GRAY)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img1)
    ax[1].imshow(img1_mos)
    ax[2].imshow(img2)
    ax[0].grid(False)
    ax[1].grid(False)
    ax[2].grid(False)
    ax[0].title.set_text('Simulated default')
    ax[1].title.set_text('Simulated demosaiced')
    ax[2].title.set_text('Measured default')
    plt.show()

    # Ensure normalization
    img1 = cv2.normalize(img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img1_arr = cv2.normalize(img1_arr, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img2_arr = cv2.normalize(img2_arr, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img1_g_2 = cv2.normalize(img1_g_2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img2_g_2 = cv2.normalize(img2_g_2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    plt.imshow(img2)
    plt.grid(False)
    plt.show()

    # Compute SSIMs
    ssim_g, _ = ssim(img1_g, img2_g, full=True, data_range=np.max(img1), channel_axis=-1)
    ssim_rgb, _ = ssim(img1_arr, img2_arr, full=True, data_range=np.max(img1), channel_axis=-1)
    ssim_g2, _ = ssim(img1_g_2, img2_g_2, full=True, data_range=np.max(img1), channel_axis=-1)
    print(f'SSIM Greyscale: {ssim_g}')
    print(f'SSIM RGB: {ssim_rgb}')
    print(f'SSIM Greyscale corrected: {ssim_g2}')

    if view:
        fig, axs = plt.subplots(1, 3, figsize=(10, 6))
        axs[0].imshow(img1)
        axs[1].imshow(img1_arr)
        axs[2].imshow(img2_arr)
        axs[0].grid(False)
        axs[1].grid(False)
        axs[2].grid(False)
        axs[0].title.set_text('Simulated img')
        axs[1].title.set_text('ISP model')
        axs[2].title.set_text('Measured img')

        plt.show()

        fig, axs = plt.subplots(2, 4, figsize=(10, 6))
        axs[0, 0].imshow(img1_g, cmap='gray')
        axs[0, 0].grid(False)
        axs[1, 0].imshow(img2_g, cmap='gray')
        axs[1, 0].grid(False)
        axs[0, 1].imshow(img1)
        axs[0, 1].grid(False)
        axs[1, 1].imshow(img2_arr)
        axs[1, 1].grid(False)
        axs[0, 2].imshow(img1_arr)
        axs[0, 2].grid(False)
        axs[1, 2].imshow(img2_arr)
        axs[1, 2].grid(False)
        axs[0, 3].imshow(img1_g_2, cmap='gray')
        axs[0, 3].grid(False)
        axs[1, 3].imshow(img2_g_2, cmap='gray')
        axs[1, 3].grid(False)

        # Show
        plt.suptitle(f'Comparison between simulated and measured images')
        axs[0, 0].title.set_text('Simulated (greyscale)')
        axs[1, 0].title.set_text('Measured (greyscale)')
        axs[0, 1].title.set_text('Simulated (colour)')
        axs[1, 1].title.set_text('Measured (colour)')
        axs[0, 2].title.set_text('Simulated (basic ISP, colour)')
        axs[1, 2].title.set_text('Measured (colour)')
        axs[0, 3].title.set_text('Simulated (basic ISP, greyscale)')
        axs[1, 3].title.set_text('Measured (greyscale)')
        axs[0, 0].set_xlabel(f'SSIM: {ssim_g:.3f}')
        axs[0, 2].set_xlabel(f'SSIM: {ssim_rgb:.3f}')
        axs[0, 3].set_xlabel(f'SSIM: {ssim_g2:.3f}')
        plt.tight_layout()

        # Plot colour histograms
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        b1, g1, r1 = get_colour_histograms(img1)
        axs[0, 0].plot(r1[1][:256], r1[0], color='r')
        axs[0, 0].plot(g1[1][:256], g1[0], color='g')
        axs[0, 0].plot(b1[1][:256], b1[0], color='b')
        b1, g1, r1 = get_colour_histograms(img1_arr)
        axs[0, 1].plot(r1[1][:256], r1[0], color='r')
        axs[0, 1].plot(g1[1][:256], g1[0], color='g')
        axs[0, 1].plot(b1[1][:256], b1[0], color='b')
        b2, g2, r2 = get_colour_histograms(img2_arr)
        axs[1, 0].plot(r2[1][:256], r2[0], color='r')
        axs[1, 0].plot(g2[1][:256], g2[0], color='g')
        axs[1, 0].plot(b2[1][:256], b2[0], color='b')
        axs[1, 1].plot(r2[1][:256], r2[0], color='r')
        axs[1, 1].plot(g2[1][:256], g2[0], color='g')
        axs[1, 1].plot(b2[1][:256], b2[0], color='b')

        plt.suptitle(f'Colour histograms')
        axs[0, 0].title.set_text('Simulated (colour bins)')
        axs[1, 0].title.set_text('Measured (colour bins)')
        axs[0, 1].title.set_text('Simulated (basic ISP, colour bins)')
        axs[1, 1].title.set_text('Measured (colour bins)')
        plt.tight_layout()
        plt.show()

        # Plot lineouts of the image
        img_len = img1_g.shape[0] // 2
        plt.figure(figsize=(10, 6))
        plt.plot(img1_g[img_len, :], '.-', label='Simulated')
        plt.plot(img1_g_2[img_len, :], '.-', label='Simulated (basic ISP)')
        plt.plot(img2_g[img_len, :], '.-', label='Measured')
        plt.title(f'Lineout for row {img_len}')
        plt.ylabel('Pixel Intensity')
        plt.xlabel('Pixel index')
        plt.legend()
        plt.show()


def evaluate_sharpness(path_s, path_m, camera='Simulated/Camera 1'):
    """Evaluate the SSIM between different specified regions to determine the effect of changing image sharpness"""
    view = False

    sharpnesses = ['02', '03', '04', '05', '06', '08', '1']
    targets = [c1_holey_w, c1_holey_b, c1_usaf, c1_h_mannequin, c1_grass, c1_sky]
    labels = ['Holey Target (White)', 'Holey Target (Black)', 'USAF Target', 'Heated Mannequin', 'Grass', 'Overcast Sky']
    i = 0
    for target in targets:
        ssim_values = []
        for sharpness in sharpnesses:
            path_s = f'C:/Users/ElliotLondon/Documents/PythonLocal/S4CSCardington/data/{camera}/sim_2022-03-05-10-02-36_f_250_sh{sharpness}.png'
            # if sharpness == '1':
            #     view = True
            # else:
            #     view = False
            ssim_values.append(visualise_ssim_regions([path_s, path_m], view=view, area=target))
        plt.plot([0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0], ssim_values, '.-', label=labels[i])
        i += 1
    plt.title('Image sharpness vs SSIM result')
    plt.xlabel('Image sharpness')
    plt.xlabel('SSIM Result')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Load images
    camera = 'Simulated/Camera 1'
    sim_path = f'C:/Users/ElliotLondon/Documents/PythonLocal/S4CSCardington/data/{camera}/sim_2022-03-05-10-02-36_f_25_sh1.png'
    meas_path = f'C:/Users/ElliotLondon/Documents/PythonLocal/S4CSCardington/data/{camera}/2022-03-05-10-02-36.png'

    # camera = 'Simulated/Camera 4'
    # sim_path = f'C:/Users/ElliotLondon/Documents/PythonLocal/S4CSCardington/data/{camera}/sim_2022-10-20-07-39-25_f_25_sh05.png'
    # meas_path = f'C:/Users/ElliotLondon/Documents/PythonLocal/S4CSCardington/data/{camera}/2022-10-20-07-39-25.png'

    # # Run analysis routine for the two images, selecting regions
    # for img_path in [sim_path, meas_path]:
    #     root = tk.Tk()
    #     app = ImageCropper(root, img_path)
    #     root.mainloop()
    # image_1, image_2 = read_images(sim_path, meas_path)
    # image_1, image_2 = process_images(image_1, image_2, area=all_rects)
    # ssim_index = visualise_ssim_regions(image_1, image_2, view=True, area=all_rects)

    # Compare region SSIMs
    for region in [c1_usaf]:
        image_1, image_2 = read_images(sim_path, meas_path)

        # Switch cases for quickly viewing entire image rather than regions
        img_area = 'na'
        if img_area == 'full':
            img_area = [[0, 0, image_1.shape[1], image_1.shape[0]], [0, 0, image_2.shape[1], image_2.shape[0]]]
        else:
            img_area = region

        image_1, image_2 = process_images(image_1, image_2, area=img_area)
        ssim_index = visualise_ssim_regions(image_1, image_2, view=True, area=img_area)

    # # Perform loop analysis here
    # evaluate_sharpness(sim_path, meas_path)

    print('All done!')
