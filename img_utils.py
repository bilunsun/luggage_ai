import argparse
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image
import torchvision
from tqdm.auto import tqdm


output_folder = "filtered/"


def imshow(img):
    img = img / 2 + 0.5
    img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


def show_random_images(loader):
    images, labels = next(iter(loader))
    imshow(torchvision.utils.make_grid(images))

    return labels


def filter_invalid_images(path):
    filenames = [f for f in listdir(path) if isfile(join(path, f))]
    for filename in tqdm(filenames):
        img_name = join(mypath, filename)
        try:
            img = Image.open(img_name)
            img.save(output_folder + filename)
        except:
            continue


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--folder",
                        help='Folder input path containing images that will be resized',
                        required=True,
                        type=str
                        )

    args = parser.parse_args()

    filter_invalid_images(args.folder)

