import os
import cv2

import numpy as np

from typing import List, Dict


from image_nets.configs.file_paths import DATA_DIR


class CamVidReader:

    def __init__(self, data_dir: str = os.path.join(DATA_DIR, 'camvid')):

        self.data_dir: str = data_dir
        self.classes: List[str] = ['sky', 'building', 'pole', 'road', 'pavement',
                                   'tree', 'signsymbol', 'fence', 'car',
                                   'pedestrian', 'bicyclist', 'unlabelled']
        self.class_to_ix: Dict[str, int] = {cls: ix for ix, cls in enumerate(self.classes)}

    def get_split(self, split: str):

        images_dir = os.path.join(self.data_dir, split)
        masks_dir = os.path.join(self.data_dir, f'{split}annot')

        image_ids = os.listdir(images_dir)

        for image_id in image_ids:

            # Read image
            image_path = os.path.join(images_dir, image_id)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Read mask
            mask_path = os.path.join(masks_dir, image_id)
            single_mask = cv2.imread(mask_path, 0)

            # Stack masks for different classes
            masks = [(single_mask == ix) for ix in self.class_to_ix.values()]
            mask = np.stack(masks, axis=-1).astype('float')

            yield image, mask



