
from yacs.config import CfgNode as CN
import os



project_root = os.getcwd()

_C = CN()


_C.FILE = CN()

_C.FILE.PRETRAIN_WEIGHT_ROOT = project_root+'/Weights/pretrained'  
_C.FILE.MODEL_SAVE_ROOT = project_root+'/Weights/trained'          
_C.FILE.VGG16_WEIGHT = 'vgg16_reducedfc.pth'                    

_C.DEVICE = CN()

_C.DEVICE.MAINDEVICE = 'cuda:0' 
_C.DEVICE.TRAIN_DEVICES = [0] 
_C.DEVICE.TEST_DEVICES = [0]  

_C.MODEL = CN()

_C.MODEL.INPUT = CN()
_C.MODEL.INPUT.IMAGE_SIZE = 300         
_C.MODEL.INPUT.PIXEL_MEAN = [0, 0, 0]  
_C.MODEL.INPUT.PIXEL_STD = [1, 1, 1] 

_C.MODEL.ANCHORS = CN()
_C.MODEL.ANCHORS.FEATURE_MAPS = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)] 
_C.MODEL.ANCHORS.MIN_SIZES = [30, 60, 111, 162, 213, 264]   
_C.MODEL.ANCHORS.MAX_SIZES = [60, 111, 162, 213, 264, 315] 
_C.MODEL.ANCHORS.ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]   
_C.MODEL.ANCHORS.BOXES_PER_LOCATION = [4, 6, 6, 6, 4, 4]  
_C.MODEL.ANCHORS.OUT_CHANNELS = [512, 1024, 512, 256, 256, 256]
_C.MODEL.ANCHORS.CLIP = True           
_C.MODEL.ANCHORS.THRESHOLD = 0.5        
_C.MODEL.ANCHORS.CENTER_VARIANCE = 0.1 
_C.MODEL.ANCHORS.SIZE_VARIANCE = 0.2   

_C.TRAIN = CN()

_C.TRAIN.NEG_POS_RATIO = 3      
_C.TRAIN.MAX_ITER = 1200      
_C.TRAIN.BATCH_SIZE = 10        
_C.TRAIN.NUM_WORKERS = 4       
_C.OPTIM = CN()

_C.OPTIM.LR = 1e-3              
_C.OPTIM.MOMENTUM = 0.9         
_C.OPTIM.WEIGHT_DECAY = 5e-4    

_C.OPTIM.SCHEDULER = CN()       
_C.OPTIM.SCHEDULER.GAMMA = 0.1  
_C.OPTIM.SCHEDULER.LR_STEPS = [80000, 100000]


_C.MODEL.TEST = CN()

_C.MODEL.TEST.NMS_THRESHOLD = 0.45              
_C.MODEL.TEST.CONFIDENCE_THRESHOLD = 0.01       
_C.MODEL.TEST.MAX_PER_IMAGE = 100             
_C.MODEL.TEST.MAX_PER_CLASS = -1               


_C.DATA = CN()


_C.DATA.DATASET = CN()
_C.DATA.DATASET.NUM_CLASSES =21
_C.DATA.DATASET.CLASS_NAME = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

_C.DATA.DATASET.DATA_DIR = 'D:/360Downloads/SSD-Pytorch/Data/VOCdevkit/VOC2007'  
_C.DATA.DATASET.TRAIN_SPLIT = 'train'       
_C.DATA.DATASET.TEST_SPLIT = 'val'         

_C.DATA.DATALOADER = CN()


_C.STEP = CN()
_C.STEP.VIS_STEP = 10          
_C.STEP.MODEL_SAVE_STEP = 1000  
_C.STEP.EVAL_STEP = 1000        

