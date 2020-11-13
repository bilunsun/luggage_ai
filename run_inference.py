import argparse
from PIL import Image
import time
from torchvision import transforms

from neural_network import Model


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def main(img_path, n=1_000):
    jpg_img = Image.open(img_path)

    model = Model().cpu()
    model.eval()

    start_time_s = time.time()
    for i in range(n):
        tensor_img = transform(jpg_img)
        tensor_img = tensor_img.unsqueeze(0)

        preds = model(tensor_img)
    elapsed_time_s = time.time() - start_time_s
    inference_time_s = elapsed_time_s / n

    print(inference_time_s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--img_path",
                        help="Path of the image to run inference on",
                        required=True,
                        type=str)

    args = parser.parse_args()

    main(args.img_path)


"""
This script tests the inference speed of the neural network.
When using the GPU, the average inference time is around 20ms; when using the GPU, 40ms.
This is takes into account the pre-processing necessary when given an image (e.g. snapshot from a video feed), including resizing, normalizing and conversion to a three-dimensional tensor.
Of course this will depend on the specifications of the machine. 
However, this script does not aim to find out the exact inference time of the luggage machine.
Instead, it is simply to show that the time is orders of magniture smaller than our specification of 2 seconds.
Therefore, the inference speed specification is met.
"""