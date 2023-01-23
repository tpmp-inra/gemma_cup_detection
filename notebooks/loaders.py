import os
import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def load_image(file_path):
    try:
        img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(file_path)
    else:
        return img


class GemmaDataset(Dataset):
    def __init__(
        self,
        csv,
        images_path,
        transform=None,
        yxyx: bool = False,
        return_id: bool = False,
        bboxes: bool = False,
    ):
        self.boxes = csv.copy()
        self.images = list(self.boxes.filename.unique())
        self.transforms = transform
        self.images_path = images_path
        if transform is not None:
            self.width, self.height = transform[0].width, transform[0].height
        else:
            self.width, self.height = 0, 0
        self.yxyx = yxyx
        self.return_id = return_id
        self.bboxes = bboxes

    def __len__(self):
        return len(self.images)

    def load_boxes(self, idx):
        if "x" in self.boxes.columns:
            boxes = self.boxes[self.boxes.filename == self.images[idx]].dropna()
            size = boxes.shape[0]
            return (
                (size, boxes[["x1", "y1", "x2", "y2"]].values) if size > 0 else (0, [])
            )
        return 0, []

    def load_image(self, idx):
        return load_image(os.path.join(self.images_path, self.images[idx]))

    def get_by_sample_name(self, filename):
        return self[self.images.index(filename)]

    def draw_image_with_boxes(self, filename):
        image, labels, *_ = self[self.images.index(filename)]
        boxes = labels[self.get_boxes_key()]
        for box in boxes:
            box_indexes = [1, 0, 3, 2] if self.yxyx is True else [0, 1, 2, 3]
            image = cv2.rectangle(
                image,
                # Boxes are in yxyx format
                (int(box[box_indexes[0]]), int(box[box_indexes[1]])),
                (int(box[box_indexes[2]]), int(box[box_indexes[3]])),
                (255, 0, 0),
                2,
            )
        return image

    def get_boxes_key(self):
        return "bboxes" if self.bboxes is True else "boxes"

    def __getitem__(self, index):
        num_box, boxes = self.load_boxes(
            index
        )  # return list of [xmin, ymin, xmax, ymax]
        img = self.load_image(index)  # return an image

        if num_box > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            # negative example, ref: https://github.com/pytorch/vision/issues/2144
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        image_id = torch.tensor([index])
        labels = torch.ones((num_box,), dtype=torch.int64)
        target = {
            self.get_boxes_key(): boxes,
            "labels": labels,
            "image_id": image_id,
            "area": torch.as_tensor(
                (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                dtype=torch.float32,
            ),
            "iscrowd": torch.zeros((num_box,), dtype=torch.int64),
            "img_size": torch.tensor([self.height, self.width]),
            "img_scale": torch.tensor([1.0]),
        }

        if self.transforms is not None:
            sample = {
                "image": img,
                "bboxes": target[self.get_boxes_key()],
                "labels": labels,
            }
            sample = self.transforms(**sample)
            img = sample["image"]
            if num_box > 0:
                # Convert to ndarray to allow slicing
                boxes = np.array(sample["bboxes"])
                # Convert to yxyx
                if self.yxyx is True:
                    boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]
                # Convert to tensor
                target[self.get_boxes_key()] = torch.as_tensor(
                    boxes, dtype=torch.float32
                )
            else:
                target[self.get_boxes_key()] = torch.zeros((0, 4), dtype=torch.float32)
        else:
            img = transforms.ToTensor()(img)

        return img, target, image_id if self.return_id is True else img, target

    def test_image(self, filename: str, image_size):
        t = get_test_image_transform(image_size=image_size)
        i = self.load_image(idx=self.images.index(filename))
        return t(image=i)["image"]


albumentaion_p = 0.5


def build_albumentations(image_size):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    return {
        "resize": [
            A.Resize(height=image_size, width=image_size, p=1),
        ],
        "train": [
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.3,
            ),
            A.RandomGamma(p=0.3),
        ],
        "to_tensor": [A.Normalize(mean=mean, std=std, p=1), ToTensorV2()],
        "un_normalize": [
            A.Normalize(
                mean=[-m / s for m, s in zip(mean, std)],
                std=[1.0 / s for s in std],
                always_apply=True,
                max_pixel_value=1.0,
            ),
        ],
    }


def get_train_transform(image_size):
    td = build_albumentations(image_size=image_size)
    return A.Compose(
        td["resize"] + td["train"] + td["to_tensor"],
        bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
    )


def get_valid_transform(image_size):
    td = build_albumentations(image_size=image_size)
    return A.Compose(
        td["resize"] + td["to_tensor"],
        bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
    )


def get_test_image_transform(image_size):
    td = build_albumentations(image_size=image_size)
    return A.Compose(
        td["resize"] + td["train"],
        bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
    )


def get_resize_only_image_transform(image_size):
    td = build_albumentations(image_size=image_size)
    return A.Compose(
        td["resize"],
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


def _update_axis(axis, image, title=None, fontsize=10, remove_axis=True):
    axis.imshow(image, origin="upper")
    if title is not None:
        axis.set_title(title, fontsize=fontsize)


def make_patches_grid(images, row_count, col_count=None, figsize=(20, 20)):
    col_count = row_count if col_count is None else col_count
    _, axii = plt.subplots(row_count, col_count, figsize=figsize)
    for ax, image in zip(axii.reshape(-1), images):
        try:
            _update_axis(axis=ax, image=image, remove_axis=True)
        except:
            pass
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()
