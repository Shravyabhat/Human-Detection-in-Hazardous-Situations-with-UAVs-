import os
import cv2
import numpy as np
from PIL import Image
import skimage.io as io
from guidedfiltering import guided_filter  # Assuming this is a custom function you have


class HazeRemoval(object):
    def __init__(self, omega=0.95, t0=0.1, radius=7, r=20, eps=0.001):
        self.omega = omega
        self.t0 = t0
        self.radius = radius
        self.r = r
        self.eps = eps

    def process_images(self):
        input_folder = "D:\YOLOV8_Hazardous\Dehazed\Hazy"  # Update with your input folder path
        output_folder = "D:\YOLOV8_Hazardous\Dehazed"  # Update with your output folder path

        input_folder = os.path.abspath(input_folder)
        output_folder = os.path.abspath(output_folder)

        # Create output folders if they don't exist
        dark_folder = os.path.join(output_folder, "dark_prior")
        trans_folder = os.path.join(output_folder, "transmission")
        final_folder = os.path.join(output_folder, "final_result")
        guided_folder = os.path.join(output_folder, "guided_filter")

        os.makedirs(dark_folder, exist_ok=True)
        os.makedirs(trans_folder, exist_ok=True)
        os.makedirs(final_folder, exist_ok=True)
        
        os.makedirs(guided_folder, exist_ok=True)

        # Process each image in the input folder
        for filename in os.listdir(input_folder):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                input_path = os.path.join(input_folder, filename)
                self.open_image(input_path)
                self.get_dark_channel()
                self.get_air_light()
                self.get_transmission()
                self.guided_filter()
                self.recover()

                # Save output images with the same filename as input
                output_name = os.path.splitext(filename)[0]
                cv2.imwrite(os.path.join(dark_folder, f"{output_name}_dark.jpg"), (self.dark * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(trans_folder, f"{output_name}_transmission.jpg"), (self.tran * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(final_folder, f"{output_name}_final.jpg"), self.dst[:, :, ::-1])  # BGR to RGB

                # Save guided filter transmission map
                cv2.imwrite(os.path.join(guided_folder, f"{output_name}_guided.jpg"), (self.gtran * 255).astype(np.uint8))

    def open_image(self, img_path):
        img = Image.open(img_path)
        self.src = np.array(img).astype(np.double) / 255.
        self.rows, self.cols, _ = self.src.shape
        self.dark = np.zeros((self.rows, self.cols), dtype=np.double)
        self.Alight = np.zeros((3), dtype=np.double)
        self.tran = np.zeros((self.rows, self.cols), dtype=np.double)
        self.dst = np.zeros_like(self.src, dtype=np.double)

    def get_dark_channel(self):
        tmp = self.src.min(axis=2)
        for i in range(self.rows):
            for j in range(self.cols):
                rmin = max(0, i - self.radius)
                rmax = min(i + self.radius, self.rows - 1)
                cmin = max(0, j - self.radius)
                cmax = min(j + self.radius, self.cols - 1)
                self.dark[i, j] = tmp[rmin:rmax + 1, cmin:cmax + 1].min()

    def get_air_light(self):
        flat = self.dark.flatten()
        flat.sort()
        num = int(self.rows * self.cols * 0.001)
        threshold = flat[-num]
        tmp = self.src[self.dark >= threshold]
        tmp.sort(axis=0)
        self.Alight = tmp[-num:, :].mean(axis=0)

    def get_transmission(self):
        for i in range(self.rows):
            for j in range(self.cols):
                rmin = max(0, i - self.radius)
                rmax = min(i + self.radius, self.rows - 1)
                cmin = max(0, j - self.radius)
                cmax = min(j + self.radius, self.cols - 1)
                pixel = (self.src[rmin:rmax + 1, cmin:cmax + 1] / self.Alight).min()
                self.tran[i, j] = 1. - self.omega * pixel

    def guided_filter(self):
        self.gtran = guided_filter(self.src, self.tran, self.r, self.eps)

    def recover(self):
        self.gtran[self.gtran < self.t0] = self.t0
        t = self.gtran.reshape(*self.gtran.shape, 1).repeat(3, axis=2)
        self.dst = (self.src - self.Alight) / t + self.Alight
        self.dst *= 255
        self.dst[self.dst > 255] = 255
        self.dst[self.dst < 0] = 0
        self.dst = self.dst.astype(np.uint8)


if __name__ == '__main__':
    hr = HazeRemoval()
    hr.process_images()
