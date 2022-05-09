import os
import numpy as np
import cv2

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class GemmaDataset(Dataset):
    def __init__(self, csv, images_path, transform=None, get_item_with_index: bool = True):
        self.boxes = csv.copy()
        self.images = list(self.boxes.filename.unique())
        self.transforms = transform
        self.images_path = images_path
        self.get_item_with_index: bool = get_item_with_index

    def __len__(self):
        return len(self.images)

    def load_boxes(self, idx):
        if "x" in self.boxes.columns:
            boxes = self.boxes[self.boxes.filename == self.images[idx]].dropna()
            size = boxes.shape[0]
            if size > 0:
                boxes = boxes[["x", "y", "width", "height"]].values
                boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
                boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
                return size, boxes
            return 0, []
        return 0, []

    def load_image(self, idx):
        image = cv2.cvtColor(
            cv2.imread(
                os.path.join(
                    self.images_path, 
                    self.images[idx]
                ), 
                cv2.IMREAD_COLOR
            ), 
            cv2.COLOR_BGR2RGB
        ).astype(np.float32)
        image /= 255.0
        return image

    def get_by_sample_name(self, filename):
        return self[self.images.index(filename)]

    def __getitem__(self, index):
        num_box, boxes = self.load_boxes(index)  # return list of [xmin, ymin, xmax, ymax]
        img = self.load_image(index)  # return an image

        if num_box > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            # negative example, ref: https://github.com/pytorch/vision/issues/2144
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        image_id = torch.tensor([index])
        labels = torch.ones((num_box,), dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": torch.as_tensor(
                (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                dtype=torch.float32,
            ),
            "iscrowd": torch.zeros((num_box,), dtype=torch.int64),
        }

        if self.transforms is not None:
            sample = {"image": img, "bboxes": target["boxes"], "labels": labels}
            sample = self.transforms(**sample)
            img = sample["image"]
            if num_box > 0:
                target["boxes"] = torch.stack(
                    tuple(map(torch.tensor, zip(*sample["bboxes"])))
                ).permute(1, 0)
            else:
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        else:
            img = transforms.ToTensor()(img)

        if self.get_item_with_index is True:
            return img, target, image_id
        else:
            return img, target


albumentaion_p = 0.5


def get_train_transform():
    return A.Compose(
        [
            A.Flip(p=albumentaion_p),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=albumentaion_p,
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            #         A.FancyPCA(p=albumentaion_p),
            A.RandomRotate90(p=albumentaion_p),
            ToTensorV2(p=1.0),
        ],
        bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
    )


def get_valid_transform():
    return A.Compose(
        [ToTensorV2(p=1.0)],
        bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
    )


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append(
            "{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3])
        )

    return " ".join(pred_strings)


def show_predictions(images, targets, image_ids, device, results):
    images = list(image.to(device) for image in images)
    sample = images[0].permute(1, 2, 0).cpu().numpy()

    boxes = results[image_ids[0].cpu().numpy()[0]]["boxes"]
    scores = results[image_ids[0].cpu().numpy()[0]]["scores"]

    _, ax = plt.subplots(1, 1, figsize=(16, 8))

    for box, score in zip(boxes, scores):
        cv2.rectangle(
            sample,
            (box[0], box[1]),
            (box[2], box[3]),
            (round((1 - score) * 255), round(score * 255), 0),
            2,
        )

    ax.set_axis_off()
    ax.imshow(sample)


def predict_loader(model, loader, device, detection_threshold):
    results = {}
    model.eval()

    for images, _, image_ids in loader:

        images = list(image.to(device) for image in images)
        outputs = model(images)

        for i, _ in enumerate(images):

            boxes = outputs[i]["boxes"].data.cpu().numpy()
            scores = outputs[i]["scores"].data.cpu().numpy()

            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            scores = scores[scores >= detection_threshold]
            image_id = image_ids[i]

            results[image_id.cpu().numpy()[0]] = {
                "image_id": image_id.cpu().numpy()[0],
                "PredictionString": format_prediction_string(boxes, scores),
                "scores": scores,
                "boxes": boxes,
            }

    return results


if __name__ == "__main__":
    a = get_train_transform()