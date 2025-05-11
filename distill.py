import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.distill import DetectionDistiller
from ultralytics.models.yolo.segment.distill import SegmentationDistiller
from ultralytics.models.yolo.pose.distill import PoseDistiller
from ultralytics.models.yolo.obb.distill import OBBDistiller

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': '/root/autodl-tmp/v8-cl/old.pt',
        'data':'dataset/new.yaml',
        'imgsz': 640,
        'epochs': 300,
        'batch': 24,
        'workers': 8,
        'cache': False,
        'pretrained': True,
        'optimizer': 'SGD',
        'freeze': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'device': '0',
        'close_mosaic': 20,
        'project':'runs/distill',
        'name':'l1',
        
        # distill
        'prune_model': False,
        'teacher_weights': '/root/autodl-tmp/v8-cl/old.pt',
        'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8l.yaml',
        'kd_loss_type': 'all',
        'kd_loss_decay': 'cosine_epoch',
        
        'logical_loss_type': 'l1',
        'logical_loss_ratio': 1,
        
        'teacher_kd_layers': '15,18,21',
        'student_kd_layers': '15,18,21',
        'feature_loss_type': 'cwd',
        'feature_loss_ratio': 1
    }
    
    model = DetectionDistiller(overrides=param_dict)
    # model = SegmentationDistiller(overrides=param_dict)
    # model = PoseDistiller(overrides=param_dict)
    # model = OBBDistiller(overrides=param_dict)
    model.distill()