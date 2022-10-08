import os
import bisect
from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
import openslide


def getFgROI(colorimg):
    rgb_max = np.max(colorimg, axis=2)
    rgb_min = np.min(colorimg, axis=2)
    rgb_diff = cv2.GaussianBlur(rgb_max - rgb_min, (5, 5), 0)
    thresh_bin = rgb_diff.max() / 5
    mask_bin = np.where(rgb_diff > 10, 1, 0)
    return mask_bin

# crop at level 0
class BaseReader(metaclass=ABCMeta):
    """Slide Reader base"""

    def __init__(self, slide_file, params):
        self.slide_file = slide_file
        self.params = params
        self.slide_id = os.path.basename(self.slide_file).split('.')[0]

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def crop_patch(self, x, y):
        pass

    @property
    def ratio(self):
        return self._ratio
    
    @property
    def width(self):
        return self._width
    
    @property
    def height(self):
        return self._height

    @property
    def slide_pixel_size(self):
        return self._slide_pixel_size
    
    def get_crop_region(self):
        crop_size_h = self.params["crop_size_h"]
        crop_size_w = self.params["crop_size_w"]
        crop_overlap = self.params["crop_overlap"]

        x_min, x_max = 0, self.width
        y_min, y_max = 0, self.height

        crop_size_w_ = int(crop_size_w * self.ratio)
        crop_size_h_ = int(crop_size_h * self.ratio)
        crop_overlap_ = int(crop_overlap * self.ratio)

        crop_step_x = (crop_size_w_ - crop_overlap_)
        crop_step_y = (crop_size_h_ - crop_overlap_)
            
        xs = np.arange(x_min, x_max - crop_size_w_, crop_step_x)
        ys = np.arange(y_min, y_max - crop_size_h_, crop_step_y)
        
        # crop path uses these property
        self.crop_size_w = crop_size_w
        self.crop_size_h = crop_size_h
        self.crop_size_w_ = crop_size_w_
        self.crop_size_h_ = crop_size_h_

        region_x, region_y = np.meshgrid(xs, ys)
        region = np.stack([region_x.reshape(-1), region_y.reshape(-1)], 1)
        return region


class OpenReader(BaseReader):
    """Openslide Reader"""

    def __init__(self, slide_file, params):
        super().__init__(slide_file, params)
        
        self.slide_id = os.path.basename(self.slide_file).split('.')[0]
        self.slide = openslide.OpenSlide(self.slide_file)
        self._slide_pixel_size, _ = self.get_slide_pixel_size(self.slide)
        
        self._width, self._height = self.slide.dimensions
        self._ratio = self.params["crop_pixel_size"] / self._slide_pixel_size
        
    def close(self):
        self.slide.close()

    def crop_patch(self, x, y):
        img = self.slide.read_region(level=0, location=(x, y),
                            size=(self.crop_size_w_, self.crop_size_h_))
        img = np.array(img, dtype=np.uint8)[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
        ## hard code to remove background image
        if np.mean(getFgROI(img)) < 0.01:
            return [(None, None)]
        
        img = cv2.resize(img, (self.crop_size_w, self.crop_size_h))
        patch_id = "{}_{}_{}.png".format(self.slide_id, str(x), str(y))
        return [(patch_id, img),]

    def get_slide_pixel_size(self, slide):
        """Get the pixel size of the slide"""
        try:
            pixel_size_x = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
            pixel_size_y = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])
        except Exception as e:
            pixel_size_x = 0.26
            pixel_size_y = 0.26

        return pixel_size_x, pixel_size_y

if __name__ == "__main__":
    params = {
        "crop_pixel_size": 0.31,
        "crop_size_h": 1280,
        "crop_size_w": 1280,
        "crop_overlap": 64,
    }

    slide_file = ""
    svs_slide_file = ''
    reader = OpenReader(svs_slide_file, params)

    region = reader.get_crop_region()
    
    for x, y in region:
        imgs = reader.crop_patch(x, y)
        if imgs[0][1] is not None:
            print(imgs[0][1].shape)
            cv2.imwrite("test.png", imgs[0][1])


