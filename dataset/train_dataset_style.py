import json
import cv2
import numpy as np
from torch.utils.data import Dataset
import random
from PIL import Image, ImageOps, ImageChops

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

        target = cv2.resize(cv2.imread(target_filename), (self.imagesize, self.imagesize))

        # Do not forget that OpenCV read images in BGR order.
        if 'handprint_scan' in target_filename:
            target = Image.fromarray(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
            target = self.transform(target)
        else:
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize target images to [-1, 1].
        target = (target / 127.5) - 1.0

        return dict(jpg=target, txt=prompt)

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

        #inpaint
        mask = np.ones((256+2*multiple_10, 256+2*multiple_10, 1), np.uint8) * 255
        mask[multiple_10:multiple_10+self.size[0], multiple_10:multiple_10+self.size[0]] = 0
        img = cv2.inpaint(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR), mask, 3, cv2.INPAINT_TELEA)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

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

if __name__=="__main__":
    img = Image.open('/Data_PHD/phd19_jing_li/datasets/Oracle/Oracle-241/scan/001000/001000_d00039_1.bmp')
    transform = resizeNormalize((256, 256))
    img1, img2 = transform(img, img)
    cv2.imwrite("img.bmp", img1)
    cv2.imwrite("img2.bmp", img2)




