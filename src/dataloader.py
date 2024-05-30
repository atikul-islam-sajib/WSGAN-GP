import os
import cv2
import zipfile
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import config, dump, load
from torch.utils.data import DataLoader


class Loader:
    def __init__(
        self, image_path=None, image_size=64, channels=3, split_size=0.30, batch_size=8
    ):
        self.image_path = image_path
        self.image_size = image_size
        self.channels = channels
        self.split_size = split_size
        self.batch_size = batch_size

        self.images = []

        self.RAW_DATA_PATH = config()["path"]["RAW_DATA_PATH"]
        self.PROCESSED_PATH = config()["path"]["PROCESSED_DATA_PATH"]

    def unzip_folder(self):
        if os.path.exists(self.RAW_DATA_PATH):
            with zipfile.ZipFile(self.image_path, "r") as zip_ref:
                zip_ref.extractall(self.RAW_DATA_PATH)

            print("Unzip is done in the path {}".format(self.RAW_DATA_PATH))

        else:
            raise KeyError("Raw Path does not exist in the config file".capitalize())

    def _transform(self):
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size), Image.BICUBIC),
                transforms.CenterCrop((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def _split_images(self, images=None):
        if isinstance(images, list) and (images is not None):
            test_dataset = images[0 : int(len(images) * self.split_size)]
            train_dataset = images[int(len(images) * self.split_size) :]

            return {"train": train_dataset, "test": test_dataset}

        else:
            raise ValueError("Image is not a list".capitalize())

    def _feature_extractor(self):

        self.DIRECTORY = os.path.join(self.RAW_DATA_PATH, "dataset")

        if len(os.listdir(self.DIRECTORY)) > 0:

            for image in os.listdir(self.DIRECTORY):
                if image is not None:
                    image_path = os.path.join(self.DIRECTORY, image)

                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)

                    image = self._transform()(image)

                    self.images.append(image)

                else:
                    raise ValueError(
                        "Image is not found in the datset folder: {}".format(
                            self.DIRECTORY
                        )
                    )

            dataset = self._split_images(images=self.images)

            return dataset

        else:
            print("Image is not found in the datset folder: {}".format(self.DIRECTORY))

    def create_dataloader(self):

        dataset = self._feature_extractor()

        train_dataloader = DataLoader(
            list(dataset["train"]), batch_size=self.batch_size, shuffle=True
        )

        test_dataloader = DataLoader(
            list(dataset["test"]), batch_size=self.batch_size * 2, shuffle=True
        )

        for filename, value in [
            ("train_dataloader", train_dataloader),
            ("test_dataloader", test_dataloader),
        ]:
            dump(
                value=value,
                filename=os.path.join(self.PROCESSED_PATH, filename + ".pkl"),
            )

        print(
            "train and test dataloader saved in the path {}".format(self.PROCESSED_PATH)
        )

    @staticmethod
    def plot_images():
        plt.figure(figsize=(20, 10))

        images = next(
            iter(
                load(
                    filename=os.path.join(
                        config()["path"]["PROCESSED_DATA_PATH"], "test_dataloader.pkl"
                    )
                )
            )
        )

        for index, image in enumerate(images):
            image = image.permute(1, 2, 0).detach().numpy()
            image = (image - image.min()) / (image.max() - image.min())

            plt.subplot(4, 4, index + 1)
            plt.imshow(image)
            plt.axis("off")
            plt.title("Image {}".format(index + 1))

        plt.tight_layout()
        plt.savefig(os.path.join(config()["path"]["FILES_PATH"], "images.png"))
        plt.show()

        print("Images saved in the path {}".format(config()["path"]["FILES_PATH"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Define the DataLoader".title())
    parser.add_argument(
        "--image_path",
        type=str,
        default="./data/raw/dataset.zip",
        help="Define the image path",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=config()["dataloader"]["image_size"],
        help="Define the image size".capitalize(),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config()["dataloader"]["batch_size"],
        help="Define the batch size".capitalize(),
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=config()["dataloader"]["channels"],
        help="Define the channels".capitalize(),
    )
    parser.add_argument(
        "--split_size",
        type=float,
        default=config()["dataloader"]["split_size"],
        help="Define the split size".capitalize(),
    )

    args = parser.parse_args()

    loader = Loader(
        image_path=args.image_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        channels=args.channels,
        split_size=args.split_size,
    )

    loader.unzip_folder()
    loader.create_dataloader()
    loader.plot_images()
