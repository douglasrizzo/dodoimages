#!/usr/bin/env python3.6
# coding: utf-8

import os
from multiprocessing import Pool

from PIL import Image, ImageChops, ImageOps
from scipy.special import binom
from skimage.io import imread
from skimage.measure import compare_ssim
from skimage.transform import resize
from tqdm import tqdm


def remove_border(filepath):
    """Removes monochromatic borders from pictures. Borders are detected by
       getting the color of the pixel in position (0,0) of the image, tracing a
       bounding box around the image using the extracted color as a threshold
       and cropping the outside of the bounding box.

       Image is overwritten after being cropped.

       :param filepath: the path to the image"""
    try:
        im = Image.open(filepath)
        bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            im_area = im.size[0] * im.size[1]
            bbox_proportion = bbox_area / im_area
            if bbox_proportion < .999:
                directory = os.path.dirname(os.path.abspath(filepath))
                new_image = os.path.basename(filepath)
                # new_image, file_extension = os.path.splitext(new_image)
                new_image_name = directory + '/' + new_image
                im.crop(bbox).save(new_image_name)
    except:
        print('Could not load {}'.format(filepath))


def delete_corrupt(image_paths):
    """Deletes corrupted images. Works especially well with images that
       were half-downloaded from the internet.

       :param image_paths: a list containing the paths of the images to be processed"""
    for i in image_paths:
        with Image.open(i) as im:
            try:
                im.load()
            except OSError:
                os.remove(i)


def remove_borders(image_paths):
    """Removes monochromatic borders from pictures. Borders are detected by
       getting the color of the pixel in position (0,0) of the image, tracing a
       bounding box around the image using the extracted color as a threshold
       and cropping the outside of the bounding box.

       Images are overwritten after being cropped.

       :param image_paths: a list containing the paths of the images to be processed"""

    with Pool(os.cpu_count()) as pool:
        # how to make a progressbar when using multiprocessing
        # for some reason the cast to list() is necessary,
        # as well as passing file size
        list(
            tqdm(
                pool.imap(remove_border, image_paths),
                total=len(image_paths),
                desc="Trimming borders"))

    # for x in tqdm(files, desc='Trimming borders'):


def remove_duplicates(image_paths, threshold=.8):
    """Compares the contents of images, searching for duplicates and removes one of them.
       Image similarity is calculated by using skimage.measure.compare_ssim and images
       with similarity >= threshold are deleted.

       The deleted file is the one of lowest resolution.

       :param image_paths: a list containing the paths of the images to be compared
       :param threshold: the similarity threshold after which two images are considered equal."""
    sorted(image_paths)
    data = [
        resize(imread(p, as_gray=True), (100, 100))
        for p in tqdm(image_paths, desc='Loading images')
    ]

    image_paths, data = zip(
        *sorted(zip(image_paths, data), key=lambda x: x[1][0, 0]))
    image_paths, data = list(image_paths), list(data)

    pbar = tqdm(
        total=int(binom(len(image_paths), 2)), desc='Searching for duplicates')

    while len(image_paths) > 0:
        p1 = image_paths[0]
        im1 = data[0]

        for p2, im2 in zip(image_paths[1:], data[1:]):
            pbar.update()
            if abs(im1[0, 0] - im2[0, 0]) > .1:
                break
            if compare_ssim(im1, im2) > threshold:
                rgb1 = imread(p1)
                rgb2 = imread(p2)

                if rgb1.size > rgb2.size:
                    image_paths[0] = p2
                    data[0] = im2

                os.remove(p1)
                image_paths = image_paths[1:]
                data = data[1:]
                pbar = tqdm(
                    total=int(binom(len(image_paths), 2)),
                    desc='Searching for duplicates')
                break
        image_paths = image_paths[1:]
        data = data[1:]
        del p1, im1


def _find_largest_dim(image_paths):
    max_dim = 0
    for p in tqdm(image_paths, desc='Loading images'):
        with Image.open(p) as im:
            max_dim = max(max_dim, im.size[0], im.size[1])
    return max_dim


def add_borders(image_paths):
    """Adds black borders to images, in order to make all of them the same size as the largest image.

       :param image_paths: a list containing the paths of the images to be processed
    """
    max_dim = _find_largest_dim(image_paths)

    for p in tqdm(image_paths, desc="Adding borders"):
        with Image.open(p) as im:
            old_w, old_h = im.size

            if old_h == max_dim and old_w == max_dim:
                continue

            # Set number of pixels to expand to the left, top, right,
            # and bottom, making sure to account for even or odd numbers
            add_top = add_bottom = (max_dim - old_h) // 2
            if old_h % 2 != 0:
                add_bottom += 1

            add_left = add_right = (max_dim - old_w) // 2
            if old_w % 2 != 0:
                add_right += 1

            left = int(add_left)
            top = int(add_top)
            right = int(add_right)
            bottom = int(add_bottom)

            # By default, the added pixels are black
            new_im = ImageOps.expand(im, border=(left, top, right, bottom))
            new_im.save(p)


def smart_scale(image_paths):
    max_dim = _find_largest_dim(image_paths)

    for p in tqdm(image_paths, desc="Resizing"):
        with Image.open(p) as im:
            old_w, old_h = im.size

            if old_h == max_dim or old_w == max_dim:
                continue

            if old_h >= old_w:
                new_h = max_dim
                new_w = int(old_w * max_dim / old_h)
            else:
                new_w = max_dim
                new_h = int(old_h * max_dim / old_w)

            # By default, the added pixels are black
            new_im = im.resize((new_w, new_h))
            new_im.save(p)
