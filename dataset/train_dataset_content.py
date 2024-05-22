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
        if 'source' in item.keys():
            source_filename = item['source']
        else:
            source_filename = item['target']

        target = cv2.resize(cv2.imread(target_filename), (self.imagesize, self.imagesize))
        source = cv2.resize(cv2.imread(source_filename), (self.imagesize, self.imagesize))

        if 'handprint_scan' in target_filename:
            target = Image.fromarray(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
            source = Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
            target, source = self.transform(target, source)
        else:
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, img2):

        #preparation
        multiple = self.size[0] / 64
        multiple_10 = round(10 * multiple)
        multiple_8 = round(8 * multiple)
        multiple_5 = round(5 * multiple)

        img = img.resize(self.size, self.interpolation)
        img2 = img2.resize(self.size, self.interpolation)
        ##padding
        # img = ImageOps.invert(img)
        img = ImageOps.expand(img, multiple_10)
        img2 = ImageOps.expand(img2, multiple_10)

        #inpaint
        mask = np.ones((256+2*multiple_10, 256+2*multiple_10, 1), np.uint8) * 255
        mask[multiple_10:multiple_10+self.size[0], multiple_10:multiple_10+self.size[0]] = 0
        img = cv2.inpaint(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR), mask, 3, cv2.INPAINT_TELEA)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img2 = cv2.inpaint(cv2.cvtColor(np.asarray(img2), cv2.COLOR_RGB2BGR), mask, 3, cv2.INPAINT_TELEA)
        img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

        ##rotate
        degree = random.randint(-15, 15)
        img = img.rotate(degree)
        img2 = img2.rotate(degree)
        ##displacement
        dx = random.randint(-multiple_5, multiple_5)
        dy = random.randint(-multiple_5, multiple_5)
        img = ImageChops.offset(img, dx, dy)
        img = np.asarray(img)
        img2 = ImageChops.offset(img2, dx, dy)
        img2 = np.asarray(img2)

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
        img2 = cv2.warpPerspective(img2, M, self.size)
        return img, img2

if __name__=="__main__":
    img1 = Image.open('/Data_PHD/phd19_jing_li/CUT/scan_handprint_bs8_batchnorm_whiteblack/test_latest/images/fake_B/001000_d00119 a_1.bmp')
    #erosion+dilation
    img1 = cv2.cvtColor(np.asarray(img1), cv2.COLOR_BGR2RGB)
    ret, img1 = cv2.threshold(src=img1,
                                  thresh=1,
                                  maxval=255,
                                  type=cv2.THRESH_BINARY)
    cv2.imwrite("img12.bmp", img1)
    kernel = np.ones((5, 5), np.uint8)
    img1 = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)
    # img1 = cv2.erode(img1, kernel, iterations=1)
    # cv2.imwrite("img13.bmp", img1)
    # img1 = cv2.dilate(img1, kernel, iterations=1)
    cv2.imwrite("img14.bmp", img1)

    img2 = cv2.resize(cv2.imread('/Data_PHD/phd19_jing_li/datasets/Oracle/Oracle-241/handprint/001000-001262/001000/001000b00113.bmp'), (256, 256))
    img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    #transform+erosion+dilation
    transform = resizeNormalize((256, 256))
    img2, _ = transform(img2, img2)
    cv2.imwrite("img21.bmp", img2)
    ret, img2 = cv2.threshold(src=img2,
                                  thresh=1,
                                  maxval=255,
                                  type=cv2.THRESH_BINARY)
    cv2.imwrite("img22.bmp", img2)
    kernel = np.ones((5, 5), np.uint8)
    img2 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
    # img2 = cv2.erode(img2, kernel, iterations=1)
    # cv2.imwrite("img23.bmp", img2)
    # img2 = cv2.dilate(img2, kernel, iterations=1)
    cv2.imwrite("img24.bmp", img2)






