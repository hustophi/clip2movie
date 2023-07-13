import mmcv
import numpy as np


class Resize(object):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:
    - `ratio_range` is not None: randomly sample a ratio from the ratio range
        and multiply it with the image scale.
    - `ratio_range` is None and `multiscale_mode` == "range": randomly sample a
        scale from the a range.
    - `ratio_range` is None and `multiscale_mode` == "value": randomly sample a
        scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self, img_scale, keep_ratio=False):
        self.img_scale = img_scale
        self.keep_ratio = keep_ratio

    def __call__(self, img):
        if self.keep_ratio:
            img = mmcv.imrescale(img, self.img_scale, return_scale=False)
        else:
            img = mmcv.imresize(img, self.img_scale, return_scale=False)
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(img_scale={}, keep_ratio={})').format(
            self.img_scale, self.keep_ratio)
        return repr_str


class Normalize(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, img):
        img = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, std={}, to_rgb={})'.format(
            self.mean, self.std, self.to_rgb)
        return repr_str


class CenterCrop(object):
    """Center Crop the image.

    Args:
        img_scale (tuple[int]): (w, h), Images scales for cropping.
    """

    def __init__(self, img_scale):
        self.img_scale = img_scale

    def __call__(self, img):
        h, w = img.shape[:2]
        assert w >= self.img_scale[0] and h >= self.img_scale[1]
        x1 = w // 2 - self.img_scale[0] // 2
        x2 = x1 + self.img_scale[0]-1
        y1 = h // 2 - self.img_scale[1] // 2
        y2 = x1 + self.img_scale[1]-1
        img = mmcv.imcrop(img, np.array([x1, y1, x2, y2]))
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(img_scale={})'.format(self.img_scale)
        return repr_str

class CenterCrop3D(object):
    """Center Crop the video.

    Args:
        img_scale (tuple[int]): (w, h), Images scales for cropping.
    """

    def __init__(self, img_scale):
        self.img_scale = img_scale

    def __call__(self, video):
        h, w = video.shape[2:]
        assert w >= self.img_scale[0] and h >= self.img_scale[1]
        x1 = w // 2 - self.img_scale[0] // 2
        x2 = x1 + self.img_scale[0]
        y1 = h // 2 - self.img_scale[1] // 2
        y2 = x1 + self.img_scale[1]
        #img = mmcv.imcrop(img, np.array([x1, y1, x2, y2]))
        return video[:, :, x1:x2, y1:y2]

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(img_scale={})'.format(self.img_scale)
        return repr_str

class OneImageCollate(object):
    """To collate a image.

    Args:
        deveice (int): the device to put the image
    """

    def __init__(self, device=0):
        self.device = device

    def __call__(self, img):
        img = img.unsqueeze(dim=0).cuda(self.device)
        return img

    def __repr__(self):
        return self.__class__.__name__
