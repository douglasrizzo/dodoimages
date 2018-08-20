#!/usr/bin/env python3.6
# coding: utf-8

import os
from multiprocessing import Pool

from PIL import Image, ImageChops, ImageOps
import cv2
import numpy as np
from skimage.io import imread
from skimage.measure import compare_ssim
from skimage.transform import resize
from tqdm import tqdm
from os.path import splitext


def isolate_from_uniform_background(filepath):
    im = Image.open(filepath)
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    return bbox


def remove_border(filepath):
    """Removes monochromatic borders from pictures. Borders are detected by
       getting the color of the pixel in position (0,0) of the image, tracing a
       bounding box around the image using the extracted color as a threshold
       and cropping the outside of the bounding box.

       Image is overwritten after being cropped.

       :param filepath: the path to the image"""
    try:
        im = Image.open(filepath)
        bbox = isolate_from_uniform_background(im)
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

    data = [
        resize(imread(p, as_grey=True), (100, 100))
        for p in tqdm(image_paths, desc='Loading images')
    ]

    image_paths, data = zip(
        *sorted(zip(image_paths, data), key=lambda x: x[1][0, 0]))
    image_paths, data = list(image_paths), list(data)

    pbar = tqdm(
        total=sum(range(len(image_paths))) - 1, desc='Searching for duplicates')

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
                    total=sum(range(len(image_paths))) - 1,
                    desc='Searching for duplicates')
                break
        image_paths = image_paths[1:]
        data = data[1:]
        del p1, im1


def _find_dim(image_paths: list, largest=True) -> tuple:
    w = h = 0 if largest else float('inf')
    func = max if largest else min
    for p in tqdm(image_paths, desc='Loading images'):
        with Image.open(p) as im:
            w = func(w, im.size[0])
            h = func(h, im.size[1])
    return w, h


def _add_borders(im: Image, dims: tuple) -> Image:
    """Adds black borders to an image

       :param im: a PIL.Image
       :param dims: a tuple with the new image width and height
    """
    w, h = dims
    old_w, old_h = im.size

    if old_h == h and old_w == w:
        return im

    # TODO check when dims are smaller than image dims
    background = Image.new("RGB", dims)
    background.paste(im, (int((background.size[0] - im.size[0]) / 2),
                          int((background.size[1] - im.size[1]) / 2)))

    return background


def add_borders(image_paths):
    """Adds black borders to images, in order to make all of them the same size as the largest image.

       :param image_paths: a list containing the paths of the images to be processed
    """
    max_dim = max(_find_dim(image_paths))

    for p in tqdm(image_paths, desc="Adding borders"):
        with Image.open(p) as im:
            new_im = _add_borders(im, (max_dim, max_dim))
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


def remove_background_grabcut(image_path,
                              output_path):
    img = cv2.imread(image_path)
    rect = isolate_from_uniform_background(image_path)
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    _, alpha = cv2.threshold(mask2, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(img)
    rgba = [b, g, r, alpha]
    img = cv2.merge(rgba, 4)

    # salvando como PNG
    filename, extension = splitext(output_path)
    print(filename, extension)
    if extension != '.png':
        output_path = filename + '.png'
    cv2.imwrite(output_path, img)


def remove_background_countours(image_path,
                                output_path,
                                blur_size=21,
                                canny_thresh_1=10,
                                canny_thresh_2=200,
                                mask_dilate_iter=10,
                                mask_erode_iter=10):
    # read image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # edge detection
    edges = cv2.Canny(gray, canny_thresh_1, canny_thresh_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    # find contours in edges, sort by area
    contour_info = []
    _, contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                      cv2.CHAIN_APPROX_NONE)

    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    # create empty mask, draw filled polygon on it corresponding to largest contour
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    # Smooth mask, then blur it
    mask = cv2.dilate(mask, None, iterations=mask_dilate_iter)
    mask = cv2.erode(mask, None, iterations=mask_erode_iter)
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    mask_stack = np.dstack([mask] * 3)  # create 3-channel alpha mask

    # blend masked img into MASK_COLOR background
    # use float matrices, for easy blending
    mask_stack = mask_stack.astype('float32') / 255.0
    img = img.astype('float32') / 255.0

    # dividir em rgb
    c_red, c_green, c_blue = cv2.split(img)

    # unir com a mascara das bordas
    img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))

    filename, extension = splitext(output_path)
    if extension != '.png':
        output_path = filename + '.png'
    cv2.imwrite(output_path, img_a * 255)


def remove_background(image_paths, grabcut=True):
    """
    Remove the background of a collection of images, replacing it by transparency
    :param image_paths: a list containing the paths of the images whose background will be made transparent
    :param grabcut: if True, use `cv2.grabCut` to isolate the foreground, otherwise, use contours
    """
    func = remove_background_grabcut if grabcut else remove_background_countours
    for image_path in tqdm(image_paths, desc='Removing background...'):
        func(image_path, image_path)


def composite_image(background_path, foreground_path, offset, output_path):
    """
    Pastes an image on top of another
    :param background_path: the background image
    :param foreground_path: the foreground image
    :param offset: A tuple containing the offset that the foreground image will be moved from the top of the background image's top left corner
    :param output_path: path to save the output file
    """
    background = Image.open(background_path, 'r')
    img = Image.open(foreground_path, 'r')
    bg_w, bg_h = background.size

    if not (offset[0] > bg_w or offset[1] > bg_h):
        background.paste(img, offset, img)

    filename, extension = splitext(output_path)
    if extension != '.png':
        output_path = filename + '.png'

    background.save(output_path)

