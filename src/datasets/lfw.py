from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torchvision
from PIL import Image
from torch.utils import data
from torchvision import transforms as T


class LFW(data.Dataset):
    def __init__(
        self,
        image_dir: Union[str, Path],
        metadata_filepath: Union[str, Path],
        input_shape: Tuple = (1, 128, 128),
        is_train: bool = True,
        is_shuffle: bool = True,
    ):
        image_dir = Path(image_dir) if isinstance(image_dir, str) else image_dir
        metadata_filepath = (
            Path(metadata_filepath)
            if isinstance(metadata_filepath, str)
            else metadata_filepath
        )

        with metadata_filepath.open(mode="r", encoding="utf-8") as file_obj:
            self.image_info = file_obj.readlines()
            self.image_info = list(map(lambda x: x.strip(), self.image_info))

        self.image_info = (
            np.random_permutation(self.image_info) if is_shuffle else self.image_info
        )

        normalize = T.Normalize(mean=[0.5], std=[0.5])
        self.transforms = T.Compose(
            [T.CenterCrop(input_shape[1:]), T.ToTensor(), normalize]
        )
        if is_train:
            self.transforms = T.Compose(
                [
                    T.RandomCrop(input_shape[1:]),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize,
                ]
            )

    def __getitem__(self, index):
        image_filepath, label = self.image_info[index].split(",")

        image = Image.open(image_filepath)
        image = image.convert("L")
        image = self.transforms(image)

        label = np.int32(label)
        return image.float(), label

    def __len__(self):
        return len(self.image_info)


if __name__ == "__main__":
    import cv2

    dataset = LFW_Dataset(
        image_dir="",
        metadata_filepath="",
        input_shape=(1, 128, 128),
        is_train=True,
        is_shuffle=False,
    )

    trainloader = data.DataLoader(dataset, batch_size=1)
    for i, (data, target) in enumerate(trainloader):
        # imgs, labels = data
        # print imgs.numpy().shape
        # print data.cpu().numpy()
        # if i == 0:
        img = torchvision.utils.make_grid(data).numpy()
        # print img.shape
        # print label.shape
        # chw -> hwc
        img = np.transpose(img, (1, 2, 0))
        # img *= np.array([0.229, 0.224, 0.225])
        # img += np.array([0.485, 0.456, 0.406])
        img += np.array([1, 1, 1])
        img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]

        cv2.imshow("img", img)
        cv2.waitKey()
        # break
        # dst.decode_segmap(labels.numpy()[0], plot=True)
