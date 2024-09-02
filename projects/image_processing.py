# Main file for processing image input from simulation and camera output
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageTk
import tkinter as tk
import seaborn as sns
from core.object_locations import *

sns.set(palette="dark", font_scale=1.1, color_codes=True)
sns.set_style('darkgrid', {'axes.linewidth': 1, 'axes.edgecolor': 'black'})

matplotlib.use('TkAgg')

rects = []


class ImageCropper:
    def __init__(self, root, image_path):
        self.root = root
        self.image_path = image_path
        self.rect_coords = []

        # Load the 1st image
        self.load_image()

        # Init rect variables
        self.start_x = None
        self.start_y = None
        self.rect = None

        # Keybinds
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def load_image(self):
        # Load image using PIL
        self.image = Image.open(self.image_path)
        self.tk_image = ImageTk.PhotoImage(self.image)

        # Create canvas to display the image
        self.canvas = tk.Canvas(self.root, width=self.tk_image.width(), height=self.tk_image.height())
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)

    def on_button_press(self, event):
        # Save starting coords
        self.start_x = event.x
        self.start_y = event.y

        # If rect exists, delete
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = None

    def on_mouse_drag(self, event):
        # Update rect as mouse is dragged
        cur_x, cur_y = (event.x, event.y)

        if self.rect:
            self.canvas.delete(self.rect)

        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, cur_x, cur_y, outline="red")

    def on_button_release(self, event):
        # Output coords of rect when mouse is released
        end_x, end_y = (event.x, event.y)
        print(f'Rect coords: ({self.start_x, self.start_y}), ({end_x, end_y})')

        # Normalise coords if dragged from bottom right to top left
        x1, x2, = sorted([self.start_x, end_x])
        y1, y2 = sorted([self.start_y, end_y])
        print(f'Normalised coords: x1 = {x1}, y1 = {y1}, x2 = {x2}, y2 = {y2}')

        rects.append([x1, y1, x2, y2])
        self.canvas.delete()
        self.root.destroy()


def perform_ssim_comparison(image_paths, view=True, area=None):
    # Open images and crop them based upon previously measured rects, or supplied rects
    if area is None:
        area = rects
    img1 = Image.open(image_paths[0]).crop(area[0])
    img2 = Image.open(image_paths[1]).crop(area[1])

    # Greyscale
    img1_g = np.array(img1.convert("L"))
    img2_g = np.array(img2.convert("L"))

    # Resize regions so that they match
    w1 = abs(area[0][0] - area[0][2])
    h1 = abs(area[0][1] - area[0][3])
    img2_g = cv2.resize(img2_g, (w1, h1))

    # Compute SSIM
    ssim_g, diff = ssim(img1_g, img2_g, full=True, data_range=255, channel_axis=-1)
    print(f'\nSSIM Greyscale: {ssim_g}')

    # Visualise
    visualise_ssim_regions(img1, img2, ssim_g, view=view, area=area)

    return ssim_g


def visualise_ssim_regions(img1, img2, ssim_index, view=True, area=None):
    """Tool for visualising processed SSIM regions"""
    if area is None:
        area = rects

    # Greyscale
    img1_g = np.array(img1.convert("L"))
    img2_g = np.array(img2.convert("L"))

    # Colour correction
    img1_arr = np.array(img1)
    img2_arr = np.array(img2)
    img1_arr = auto_colour_adjust(img1_arr, img2_arr)

    # Resize regions so that they match
    w1 = abs(area[0][0] - area[0][2])
    h1 = abs(area[0][1] - area[0][3])
    img2_g = cv2.resize(img2_g, (w1, h1))
    img2_arr = cv2.resize(img2_arr, (w1, h1))

    # Convert back to greyscale for comparison
    img1_g_2 = np.array(Image.fromarray(img1_arr).convert("L"))
    img2_g_2 = np.array(Image.fromarray(img2_arr).convert("L"))

    ssim_rgb, diff = ssim(img1_arr, img2_arr, full=True, data_range=255, channel_axis=-1)
    ssim_g2, diff = ssim(img1_g_2, img2_g_2, full=True, data_range=255, channel_axis=-1)
    print(f'SSIM RGB: {ssim_rgb}')
    print(f'SSIM Greyscale corrected: {ssim_g2}')

    # Plot lineouts of the image
    img_len = img1_g.shape[0] // 2
    plt.plot(img1_g[img_len, :], '.-', label='Simulated')
    plt.plot(img1_g_2[img_len, :], '.-', label='Simulated (basic ISP)')
    plt.plot(img2_g[img_len, :], '.-', label='Measured')
    plt.title(f'Lineout for row {img_len}')
    plt.ylabel('Pixel Intensity')
    plt.xlabel('Pixel index')
    plt.legend()
    plt.show()

    if view:
        fig, axs = plt.subplots(2, 4)
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

        # Plot colour histograms underneath
        b1, g1, r1 = get_colour_histograms(img1_arr)
        axs[0, 3].plot(r1[1][:256], r1[0], color='r')
        axs[0, 3].plot(g1[1][:256], g1[0], color='g')
        axs[0, 3].plot(b1[1][:256], b1[0], color='b')
        # axs[2, 0].yscale('log')

        b2, g2, r2 = get_colour_histograms(img2_arr)
        axs[1, 3].plot(r2[1][:256], r2[0], color='r')
        axs[1, 3].plot(g2[1][:256], g2[0], color='g')
        axs[1, 3].plot(b2[1][:256], b2[0], color='b')
        # axs[2, 1].yscale('log')

        # Show
        plt.suptitle(f'Comparison between simulated and measured images. SSIM: {ssim_index}')
        axs[0, 0].title.set_text('Simulated image (greyscale)')
        axs[1, 0].title.set_text('Measured image (greyscale)')
        axs[0, 1].title.set_text('Simulated image (greyscale w/ basic ISP)')
        axs[1, 1].title.set_text('Measured image (greyscale)')
        axs[0, 2].title.set_text('Simulated image (basic ISP colour bins)')
        axs[1, 2].title.set_text('Measured image (colour bins)')
        plt.tight_layout()
        plt.show()

        # # Plot next to each other for neat comparison
        # plt.figure(figsize=(15, 15))
        # plt.subplot(1, 2, 1)
        # plt.title("Simulated image RGB lineouts")
        # plt.plot(r1[1][:256], r1[0], color='r')
        # plt.plot(g1[1][:256], g1[0], color='g')
        # plt.plot(b1[1][:256], b1[0], color='b')
        #
        # plt.subplot(1, 2, 2)
        # plt.title("Measured image RGB lineouts")
        # plt.plot(r2[1][:256], r2[0], color='r')
        # plt.plot(g2[1][:256], g2[0], color='g')
        # plt.plot(b2[1][:256], b2[0], color='b')
        # plt.show()
        #
        # plt.figure(figsize=(15, 15))
        # plt.subplot(1, 2, 1)
        # plt.grid(False)
        # plt.imshow(img1_arr)
        #
        # plt.subplot(1, 2, 2)
        # plt.grid(False)
        # plt.imshow(img2_arr)
        # plt.tight_layout()
        # plt.show()


def get_colour_histograms(img_data):
    # Process image data and separate into each colour channel, enabling it to be plotted separately
    img_data = np.array(img_data)[:, :, ::-1].copy()
    red_px = np.histogram(img_data[:, :, 0], bins=256, range=[0, 255])
    green_px = np.histogram(img_data[:, :, 1], bins=256, range=[0, 255])
    blue_px = np.histogram(img_data[:, :, 2], bins=256, range=[0, 255])

    return red_px, green_px, blue_px


def colour_adjust(img_in, rgb_arr):
    """Manually adjust the colours of a cv2 uint8 image array"""
    rgb_arr = np.array(rgb_arr)

    for colour in range(0, 2):
        if rgb_arr[colour] == 0:
            continue
        elif rgb_arr[colour] < 0:
            rgb = np.zeros_like(np.array(img_in), dtype=np.uint8)
            rgb[:, :, colour] = abs(rgb_arr[colour])
            img_in = cv2.subtract(img_in, rgb)
        elif rgb_arr[colour] > 0:
            rgb = np.zeros_like(np.array(img_in), dtype=np.uint8)
            rgb[:, :, colour] = abs(rgb_arr[colour])
            img_in = cv2.add(img_in, rgb)

    return img_in


def auto_colour_adjust(img1, img2):
    """Adjust the colours of img1 to suit img2 by finding the median peaks of each channel between images"""
    # Convert images to LAB colour space and split into LAB channels
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2Lab)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2Lab)
    l1, a1, b1 = cv2.split(img1)
    l2, a2, b2 = cv2.split(img2)

    # Apply histogram matching channels
    matched_l = match_histograms(l1, l2)
    matched_a = match_histograms(a1, a2)
    matched_b = match_histograms(b1, b2)

    # Merge back together and convert back to BGR
    matched_image = cv2.merge([matched_l, matched_a, matched_b])
    matched_image = cv2.cvtColor(matched_image, cv2.COLOR_LAB2BGR)

    return matched_image


def match_histograms(source, template):
    """Match the histograms from a source to a template"""
    old_shape = source.shape
    source = source.ravel()
    template = template.ravel()

    # Get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # Calculate normalised cumulative distribution functions (CDFs)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # Use interpolation to find the pixel values in the template image which corresponds to each pixel in source
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values).astype(source.dtype)

    return interp_t_values[bin_idx].reshape(old_shape)


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
            ssim_values.append(perform_ssim_comparison([path_s, path_m], view=view, area=target))
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
    sim_path = f'C:/Users/ElliotLondon/Documents/PythonLocal/S4CSCardington/data/{camera}/sim_2022-03-05-10-02-36_f_200.png'
    meas_path = f'C:/Users/ElliotLondon/Documents/PythonLocal/S4CSCardington/data/{camera}/2022-03-05-10-02-36.png'

    # camera = 'Simulated/Camera 4'
    # sim_path = f'C:/Users/ElliotLondon/Documents/PythonLocal/S4CSCardington/data/{camera}/sim_2022-10-20-07-39-25_f_25_sh05.png'
    # meas_path = f'C:/Users/ElliotLondon/Documents/PythonLocal/S4CSCardington/data/{camera}/2022-10-20-07-39-25.png'

    # # Run analysis routine for the two images, selecting regions
    # for img_path in [sim_path, meas_path]:
    #     root = tk.Tk()
    #     app = ImageCropper(root, img_path)
    #     root.mainloop()
    # ssim_value = perform_ssim_comparison([sim_path, meas_path], view=True, area=None)

    # Compare region SSIMs
    for region in [c1_stop_sign]:
        ssim_value = perform_ssim_comparison([sim_path, meas_path], view=True, area=region)

    # # Perform loop analysis here
    # evaluate_sharpness(sim_path, meas_path)

    print('All done!')
