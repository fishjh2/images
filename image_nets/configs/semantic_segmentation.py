from dataclasses import dataclass


@dataclass
class SemanticSegmentationConfig:

    # data
    dataset: str = 'CamVid'

    # model
    encoder: str = 'se_resnext50_32x4d'
    encoder_pretrain_dataset: str = 'imagenet'

