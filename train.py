import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8l.yaml')
    #model.load('/root/autodl-tmp/v8-cl/best.pt') # loading pretrain weights
    model.train(data='/root/autodl-tmp/v8-cl/dataset/data.yaml',
                cache=True,
                imgsz=640,
                epochs=400,
                batch=24,
                close_mosaic=10,
                workers=8,
                device='0',
                pretrained=True,
                optimizer='SGD', # using SGD
                #freeze=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                patience=50,
                #lr0=0.001,
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/pest24-cl',
                name='up',
                # distill
                #prune_model=False,
                #teacher_weights='/root/autodl-tmp/v8-cl/best.pt',
                #teacher_cfg='ultralytics/cfg/models/v8/yolov8l.yaml',
                #kd_loss_type='feature',
                #kd_loss_decay='linear_epoch',

                #logical_loss_type='l2',
                #logical_loss_ratio=1,

                #teacher_kd_layers='15,18,21',
                #student_kd_layers='15,18,21',
                #feature_loss_type='cwd',
                #feature_loss_ratio=0.7

                )