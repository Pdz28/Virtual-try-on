import torch
import math
import numbers
import random
import numpy as np

from PIL import Image, ImageOps
from torchvision import transforms

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w
        self.padding = padding

    def __call__(self, sample):
        img, mask = sample['image'], sample['label']

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size # target size
        if w == tw and h == th:
            return {'image': img,
                    'label': mask}
        if w < tw or h < th:
            img = img.resize((tw, th), Image.BILINEAR)
            mask = mask.resize((tw, th), Image.NEAREST)
            return {'image': img,
                    'label': mask}

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': img,
                'label': mask}

class RandomCrop_new(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w
        self.padding = padding

    def __call__(self, sample):
        img, mask = sample['image'], sample['label']

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size # target size
        if w == tw and h == th:
            return {'image': img,
                    'label': mask}

        new_img = Image.new('RGB',(tw,th),'black')  # size is w x h; and 'white' is 255
        new_mask = Image.new('L',(tw,th),'white')  # same above

        # if w > tw or h > th
        x1 = y1 = 0
        if w > tw:
            x1 = random.randint(0,w - tw)
        if h > th:
            y1 = random.randint(0,h - th)
        # crop
        img = img.crop((x1,y1, x1 + tw, y1 + th))
        mask = mask.crop((x1,y1, x1 + tw, y1 + th))
        new_img.paste(img,(0,0))
        new_mask.paste(mask,(0,0))

        # x1 = random.randint(0, w - tw)
        # y1 = random.randint(0, h - th)
        # img = img.crop((x1, y1, x1 + tw, y1 + th))
        # mask = mask.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': new_img,
                'label': new_mask}

class Paste(object):
    def __init__(self, size,):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w

    def __call__(self, sample):
        img, mask = sample['image'], sample['label']

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size # target size
        assert (w <=tw) and (h <= th)
        if w == tw and h == th:
            return {'image': img,
                    'label': mask}

        new_img = Image.new('RGB',(tw,th),'black')  # size is w x h; and 'white' is 255
        new_mask = Image.new('L',(tw,th),'white')  # same above

        new_img.paste(img,(0,0))
        new_mask.paste(mask,(0,0))

        return {'image': new_img,
                'label': new_mask}

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': img,
                'label': mask}

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}

class HorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}

class HorizontalFlip_only_img(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}

class RandomHorizontalFlip_cihp(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # mask = Image.open()

        return {'image': img,
                'label': mask}

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}

class Normalize_255(object):
    """Normalize a tensor image with mean and standard deviation. tf use 255.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(123.15, 115.90, 103.06), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        # img = 255.0
        img -= self.mean
        img /= self.std
        img = img
        img = img[[0,3,2,1],...]
        return {'image': img,
                'label': mask}

class Normalize_xception_tf(object):
    # def __init__(self):
    #     self.rgb2bgr =

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        img = (img*2.0)/255.0 - 1
        # print(img.shape)
        # img = img[[0,3,2,1],...]
        return {'image': img,
                'label': mask}

class Normalize_xception_tf_only_img(object):
    # def __init__(self):
    #     self.rgb2bgr =

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        # mask = np.array(sample['label']).astype(np.float32)
        img = (img*2.0)/255.0 - 1
        # print(img.shape)
        # img = img[[0,3,2,1],...]
        return {'image': img,
                'label': sample['label']}

class Normalize_cityscapes(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.)):
        self.mean = mean

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        img -= self.mean
        img /= 255.0

        return {'image': img,
                'label': mask}

class ToTensor_(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.rgb2bgr = transforms.Lambda(lambda x:x[[2,1,0],...])

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(sample['image']).astype(np.float32).transpose((2, 0, 1))
        mask = np.expand_dims(np.array(sample['label']).astype(np.float32), -1).transpose((2, 0, 1))
        # mask[mask == 255] = 0

        img = torch.from_numpy(img).float()
        img = self.rgb2bgr(img)
        mask = torch.from_numpy(mask).float()


        return {'image': img,
                'label': mask}

class ToTensor_only_img(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.rgb2bgr = transforms.Lambda(lambda x:x[[2,1,0],...])

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(sample['image']).astype(np.float32).transpose((2, 0, 1))
        # mask = np.expand_dims(np.array(sample['label']).astype(np.float32), -1).transpose((2, 0, 1))
        # mask[mask == 255] = 0

        img = torch.from_numpy(img).float()
        img = self.rgb2bgr(img)
        # mask = torch.from_numpy(mask).float()


        return {'image': img,
                'label': sample['label']}

class FixedResize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}

class Keep_origin_size_Resize(object):
    def __init__(self, max_size, scale=1.0):
        self.size = tuple(reversed(max_size))  # size: (h, w)
        self.scale = scale
        self.paste = Paste(int(max_size[0]*scale))

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size
        h, w = self.size
        h = int(h*self.scale)
        w = int(w*self.scale)
        img = img.resize((h, w), Image.BILINEAR)
        mask = mask.resize((h, w), Image.NEAREST)

        return self.paste({'image': img,
                'label': mask})

class Scale(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        w, h = img.size

        if (w >= h and w == self.size[1]) or (h >= w and h == self.size[0]):
            return {'image': img,
                    'label': mask}
        oh, ow = self.size
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        return {'image': img,
                'label': mask}

class Scale_(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        w, h = img.size
        ow = int(w*self.scale)
        oh = int(h*self.scale)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        return {'image': img,
                'label': mask}

class Scale_only_img(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # assert img.size == mask.size
        w, h = img.size
        ow = int(w*self.scale)
        oh = int(h*self.scale)
        img = img.resize((ow, oh), Image.BILINEAR)
        # mask = mask.resize((ow, oh), Image.NEAREST)

        return {'image': img,
                'label': mask}

class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                img = img.resize((self.size, self.size), Image.BILINEAR)
                mask = mask.resize((self.size, self.size), Image.NEAREST)

                return {'image': img,
                        'label': mask}

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        sample = crop(scale(sample))
        return sample

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.random() * 2 * self.degree - self.degree
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}

class RandomSized_new(object):
    '''what we use is this class to aug'''
    def __init__(self, size,scale1=0.5,scale2=2):
        self.size = size
        # self.scale = Scale(self.size)
        self.crop = RandomCrop_new(self.size)
        self.small_scale = scale1
        self.big_scale = scale2

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size

        w = int(random.uniform(self.small_scale, self.big_scale) * img.size[0])
        h = int(random.uniform(self.small_scale, self.big_scale) * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)
        sample = {'image': img, 'label': mask}
        # finish resize
        return self.crop(sample)
# class Random

class RandomScale(object):
    def __init__(self, limit):
        self.limit = limit

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size

        scale = random.uniform(self.limit[0], self.limit[1])
        w = int(scale * img.size[0])
        h = int(scale * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        return {'image': img, 'label': mask}

class ResizeKeepAspectRatio(object):
    """Resize ảnh và mask mà vẫn giữ nguyên tỉ lệ khung hình (aspect ratio).
    Ảnh sẽ được scale để fit vào kích thước target, phần còn lại sẽ được pad với màu đen.
    """
    def __init__(self, size, fill_color=(0, 0, 0), fill_mask_color=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size  # (height, width)
        self.fill_color = fill_color  # Màu fill cho ảnh RGB
        self.fill_mask_color = fill_mask_color  # Màu fill cho mask

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        
        assert img.size == mask.size
        
        target_h, target_w = self.size
        original_w, original_h = img.size
        
        # Tính tỷ lệ scale để giữ nguyên aspect ratio
        scale_w = target_w / original_w
        scale_h = target_h / original_h
        scale = min(scale_w, scale_h)  # Chọn scale nhỏ hơn để đảm bảo ảnh fit trong target size
        
        # Tính kích thước mới sau khi scale
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        # Resize ảnh và mask với tỷ lệ mới
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        mask_resized = mask.resize((new_w, new_h), Image.NEAREST)
        
        # Tạo ảnh mới với kích thước target
        new_img = Image.new('RGB', (target_w, target_h), self.fill_color)
        new_mask = Image.new('L', (target_w, target_h), self.fill_mask_color)
        
        # Tính vị trí để center ảnh đã resize
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        
        # Paste ảnh đã resize vào center của ảnh mới
        new_img.paste(img_resized, (paste_x, paste_y))
        new_mask.paste(mask_resized, (paste_x, paste_y))
        
        return {'image': new_img, 'label': new_mask}

class ResizeKeepAspectRatio_only_img(object):
    """Resize chỉ ảnh mà vẫn giữ nguyên tỉ lệ khung hình, mask giữ nguyên.
    """
    def __init__(self, size, fill_color=(0, 0, 0)):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size  # (height, width)
        self.fill_color = fill_color

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        
        target_h, target_w = self.size
        original_w, original_h = img.size
        
        # Tính tỷ lệ scale để giữ nguyên aspect ratio
        scale_w = target_w / original_w
        scale_h = target_h / original_h
        scale = min(scale_w, scale_h)
        
        # Tính kích thước mới sau khi scale
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        # Resize ảnh với tỷ lệ mới
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        
        # Tạo ảnh mới với kích thước target
        new_img = Image.new('RGB', (target_w, target_h), self.fill_color)
        
        # Tính vị trí để center ảnh đã resize
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        
        # Paste ảnh đã resize vào center của ảnh mới
        new_img.paste(img_resized, (paste_x, paste_y))
        
        return {'image': new_img, 'label': mask}