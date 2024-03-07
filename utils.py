#import scipy.misc
from PIL import Image
import numpy as np
import copy
import os

class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image

def load_test_data(image_path, fine_size=256):
    img = Image.open(image_path)
    img = img.resize((fine_size*2, fine_size))
    img = np.array(img)
    # Normalize image to the range [-1, 1]
    img = img / 127.5 - 1

    return img

def check_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)

def load_train_data(image_path, load_size=286, fine_size=256, is_testing=False):
    img_A = Image.open(image_path[0])
    img_B = Image.open(image_path[1])

    if not is_testing:
        # Resize images using PIL
        img_A = img_A.resize((load_size*2, load_size))
        img_B = img_B.resize((load_size*2, load_size))

        # Random crop
        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, (load_size-fine_size)*2)))
        img_A = np.array(img_A.crop((w1, h1, w1+fine_size*2, h1+fine_size)))
        img_B = np.array(img_B.crop((w1, h1, w1+fine_size*2, h1+fine_size)))

        # Random horizontal flip
        if np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)
    else:
        # Resize images using PIL for testing
        img_A = img_A.resize((fine_size*2, fine_size))
        img_B = img_B.resize((fine_size*2, fine_size))

    # Normalize images to the range [-1, 1]
    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.

    # Concatenate images along the channel axis
    img_AB = np.concatenate((img_A, img_B), axis=2)

    return img_AB

# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale=False):
    return transform(load_image(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def load_image(path, is_grayscale=False):
    if is_grayscale:
        return np.array(Image.open(path).convert('L')).astype(np.float)
    else:
        return np.array(Image.open(path).convert('RGB')).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    # Convert images to uint8 format and adjust the range
    images = ((images + 1.0) * 127.5).astype(np.uint8)

    # Merge images
    merged_image = merge(images, size).astype(np.uint8)

    # Create a PIL Image from the numpy array
    pil_image = Image.fromarray(merged_image)

    # Save the image using PIL
    pil_image.save(path)

    return None

def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))

    # Use PIL for resizing
    cropped_image = Image.fromarray(x[j:j+crop_h, i:i+crop_w].astype(np.uint8))
    cropped_image = cropped_image.resize((resize_w, resize_h))

    return np.array(cropped_image)/127.5 - 1.

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx: # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

def norm_img(img):
    img = img / np.linalg.norm(img)
    img = (img * 2.) - 1.

    return img


def set_path(args, experiment_name):
    args.checkpoint_dir = f'./check/{experiment_name}'
    args.sample_dir = f'./check/{experiment_name}/sample'
    if args.which_direction == 'AtoB':
        args.test_dir = f'./check/{experiment_name}/testa2b'
    else:
        args.test_dir = f'./check/{experiment_name}/testb2a'
    args.conf_dir = f'./check/{experiment_name}/conf'
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    if not os.path.exists(args.conf_dir):
        os.makedirs(args.conf_dir)