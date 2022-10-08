# Abnormal cell detector

The one-stage detector RetinaNet is adopted as the abnormal cell detector to detect four abnormal cells.

## Requirements
- [detectron2](https://github.com/facebookresearch/detectron2)
- [openslide-python](https://github.com/openslide/openslide-python)

## Usage
1. train the abnormal cell detector.

you have to prepare cell-level annotated image dataset, and convert it into coco format, and regist cell dataset at [cell.py](datasets/cell.py). Then, you can train your model with the following command:

```
python train_net.py --config-file=configs/custom/retinanet_R_50_FPN.py --num-gpus=8
```

2. use the detector to detect abnormal cells (with features) of slides

you can do inference with the following command:

```
python predict_slide_feat_main.py --config-file=configs/custom/retinanet_R_50_FPN_feat.py --input=slides.csv --weights=<path>/model_final.pth
```
