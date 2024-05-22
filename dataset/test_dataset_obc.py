import json
import cv2
import numpy as np
import random
from PIL import Image
from PIL import ImageOps, ImageChops
from torch.utils.data import Dataset
from random import choice

imagenet_templates_small = [
    '{}',
]
placeholder_string = "*"

class MyDataset(Dataset):
    def __init__(self, imagesize=256, dir=None, transform=None):
        self.data = []
        self.imagesize = imagesize
        self.transform = transform
        with open(dir, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        target_filename = item['target']
        prompt = random.choice(imagenet_templates_small).format(placeholder_string)
        source_filename = item['source']
        label = item['label']

        # print(source_filename)
        source = cv2.resize(cv2.imread(source_filename), (self.imagesize, self.imagesize))
        # print(target_filename)
        target = cv2.resize(cv2.imread(target_filename), (self.imagesize, self.imagesize))

        if self.transform is not None:
            source = Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
            source = self.transform(source)
        else:
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)


        # kernel_candy = [5]
        # candy = choice(kernel_candy)
        # _, source = cv2.threshold(src=source, thresh=1, maxval=255, type=cv2.THRESH_BINARY)
        # kernel = np.ones((candy, candy), np.uint8)
        # source = cv2.morphologyEx(source, cv2.MORPH_OPEN, kernel)
        # source = cv2.dilate(source, kernel)

        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # target = to_white_on_black(target.astype(np.float32))
        # Normalize target images to [-1, 1].
        target = (target / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source, label=label)


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):

        #preparation
        multiple = self.size[0] / 64
        multiple_10 = round(10 * multiple)
        multiple_8 = round(8 * multiple)
        multiple_5 = round(5 * multiple)

        img = img.resize(self.size, self.interpolation)
        ##padding
        # img = ImageOps.invert(img)
        img = ImageOps.expand(img, multiple_10)
        ##rotate
        degree = random.randint(-15, 15)
        img = img.rotate(degree)
        ##displacement
        dx = random.randint(-multiple_5, multiple_5)
        dy = random.randint(-multiple_5, multiple_5)
        img = ImageChops.offset(img, dx, dy)
        img = np.asarray(img)

        ##randomly crop a part of img and transform it to 64*64
        pts1 = np.float32([[multiple_10 + random.randint(-multiple_8, multiple_8),
                            multiple_10 + random.randint(-multiple_8, multiple_8)],
                           [multiple_10 + random.randint(-multiple_8, multiple_8),
                            multiple_10 + self.size[0] - 1 + random.randint(-multiple_8, multiple_8)],
                           [multiple_10 + self.size[0] - 1 + random.randint(-multiple_8, multiple_8),
                            multiple_10 + random.randint(-multiple_8, multiple_8)],
                           [multiple_10 + self.size[0] - 1 + random.randint(-multiple_8, multiple_8),
                            multiple_10 + self.size[0] - 1 + random.randint(-multiple_8, multiple_8)]])
        pts2 = np.float32([[0, 0], [0, self.size[0] - 1], [self.size[0] - 1, 0], [self.size[0] - 1, self.size[0] - 1]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, self.size)
        return img
