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
        metadata_filepath: Union[str, Path],
        input_shape: Tuple = (1, 128, 128),
        is_train: bool = True,
    ):
        metadata_filepath = (
            Path(metadata_filepath)
            if isinstance(metadata_filepath, str)
            else metadata_filepath
        )

        with metadata_filepath.open(mode="r", encoding="utf-8") as file_obj:
            self.image_info = file_obj.readlines()
            self.image_info = list(map(lambda x: x.strip(), self.image_info))

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
        image_filepath, label, logist = self.image_info[index].split(",")

        image = Image.open(image_filepath)
        image = image.convert("L")
        image = self.transforms(image)

        label = np.int32(label)
        logist = np.int32(logist)
        return image.float(), label, logist

    def __len__(self):
        return len(self.image_info)


if __name__ == "__main__":
    import cv2

    dataset = LFW(
        metadata_filepath="train_data.txt",
        input_shape=(1, 128, 128),
        is_train=True,
    )

    trainloader = data.DataLoader(dataset, batch_size=1)
    for i, (data, label, logist) in enumerate(trainloader):
        img = torchvision.utils.make_grid(data).numpy()
        img = np.transpose(img, (1, 2, 0))
        img += np.array([1, 1, 1])
        img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]
        print(f"label: {label}, logist: {logist}")
        cv2.imshow("img", img)
        cv2.waitKey()
        break
