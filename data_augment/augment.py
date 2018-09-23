from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", help="path to image", required=True)
parser.add_argument("-o", "--output", help="path to output", required=True)
args = vars(parser.parse_args())

image = img_to_array(load_img(args["image"]))
image = np.expand_dims(image, axis=0)

aug = ImageDataGenerator(rotation_range=30,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode="nearest")
total = 0

image_generator = aug.flow(image, batch_size=1, save_to_dir=args["output"],
                           save_prefix="image", save_format="jpg")

for image in image_generator:
    total += 1
    if total == 10:
        break

