from detectron2.data.datasets import register_coco_instances                                                                                  

cell_train_path = ""
cell_train_json = ""
cell_val_path = ""
cell_val_json = ""

register_coco_instances("cell_train", {}, cell_train_json, cell_train_path)
register_coco_instances("cell_val", {}, cell_val_json, cell_val_path)

