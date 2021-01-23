import os
import random

import cv2
import numpy as np
from lgblkb_tools import Folder, logger
from scipy import ndimage
from skimage.exposure import match_histograms

project_folder = Folder(__file__).parent(1)
images_folder = project_folder['images']
kazakh_imgs_folder = images_folder['Kazakh']
foreign_imgs_folder = images_folder['Foreign']
feature_imgs_folder = images_folder['Features']
converted_imgs_folder = images_folder['Converted']


def jp2_to_jpg(img_path):
	"""Converts jp2 to jpg."""
	name, ext = os.path.splitext(img_path)
	
	if ext == ".jp2":
		img = cv2.imread(img_path)
		cv2.imwrite(name + ".jpg", img)


def img_info(name, img):
	"""Prints information about the image.

	Arguments
	-----------
	img:
		Image in BGR format (read by cv2 library).
	name:
		Name for the image.

	"""
	print("----------------------------------")
	print("[NAME] - %s" % name)
	print("[SHAPE] - %s" % str(img.shape))
	print("[MEAN:BLUE] - %.3f" % img[:, :, 0].mean())
	print("[MEAN:GREEN] - %.3f" % img[:, :, 1].mean())
	print("[MEAN:RED] - %.3f" % img[:, :, 2].mean())


def show_images(*args):
	"""Show all provided images one by one.

	Arguments
	-----------
	args:
		List of image objects to show.

	"""
	for i, img in enumerate(args):
		cv2.imshow("Image #{}".format(i + 1), img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


def random_resize(image, background):
	random_size = random.uniform(0.005, 0.07) * background.shape[1]
	
	h, w = image.shape[:2]
	r = random_size / float(h)
	dim = (int(w * r), int(random_size))
	dst = cv2.resize(image, dim)
	return dst


def random_rotate(image):
	"""
    Rotates an image by the given angle.

    :param image: Image to be rotated.
    :return: Rotated image.
    """
	random_angle = np.random.randint(0, 360)
	dst = ndimage.rotate(image, random_angle)
	return dst


def image_to_background(image, background, alpha):
	"""
	Randomly putting the image to the background image.

	:param image: Foreground image.
	:param background: Background image.
	:param alpha: Alpha channel extent.
	:return: Combined image with random placement of foreground image on background image and coordinates of 4 corners
	of an image along with translation coordinates x, y.
	"""
	image = random_rotate(image)
	image = random_resize(image, background)
	x = random.randint(0, background.shape[0] - image.shape[0])
	y = random.randint(0, background.shape[1] - image.shape[1])
	out = background.copy()
	beta = 1 - alpha
	
	rows, cols, channels = image.shape
	roi = out[x:x + rows, y:y + cols]
	
	img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	ret, mask = cv2.threshold(img2gray, 50, 255, cv2.THRESH_BINARY)
	mask_inv = cv2.bitwise_not(mask)
	
	img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
	img2_fg = cv2.bitwise_and(image, image, mask=mask)
	
	dst = cv2.add(img1_bg, img2_fg)
	out[x:x + rows, y:y + cols] = dst
	coordinates = [[y, x], [y + cols, x], [y + cols, x + rows], [y, x + rows]]
	
	out = cv2.addWeighted(background, alpha, out, beta, 0.0)
	
	return out, coordinates, (x, y)


@logger.trace()
def image_pipeline(img, num_of_features=100, alpha=0.3):
	logger.info(f"Number of features: {num_of_features}")
	logger.info(f"Alpha: {alpha}")
	img_copy = img.copy()
	feature_imgs = []
	
	for i in range(1, 11):
		temp_img = cv2.imread(feature_imgs_folder['feature_{}.jpg'.format(i)], -1)
		mask_img = cv2.imread(feature_imgs_folder['feature_{}.jpg'.format(i)], -1)
		black_img = np.zeros(mask_img.shape, dtype=np.uint8)
		temp_img = match_histograms(temp_img, img_copy, multichannel=True)
		temp_img = cv2.addWeighted(mask_img, 0.0, temp_img, 1.0, 0.0)
		temp_img = np.where(mask_img > 10, temp_img, black_img)
		# show_images(mask_img, white_img, temp_img)
		feature_imgs.append(temp_img)
	
	for i in range(num_of_features):
		j = np.random.randint(0, 7)
		feature_img = feature_imgs[j]
		img_copy, _, _ = image_to_background(feature_img, img_copy, alpha=alpha)
	
	return img_copy


if __name__ == "__main__":
	kazakh_img = cv2.imread(kazakh_imgs_folder['01.jpg'])
	
	for i, file in enumerate(os.listdir(foreign_imgs_folder.path)):
		if file.endswith(".jp2"):
			continue
		foreign_img = cv2.imread(os.path.join(foreign_imgs_folder.path, file))
		merge_img = image_pipeline(foreign_img, num_of_features=np.random.randint(10, 50), alpha=0.0)
		index = int(file[:2])
		if index <= 20:
			result_img = match_histograms(merge_img, kazakh_img, multichannel=True)
		cv2.imwrite(os.path.join(converted_imgs_folder.path, file), result_img)
		print("Finished processing:", i + 1, "images")
