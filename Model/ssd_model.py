
import torch
from torch import nn
from .base_models import vgg
from .structs import predictor,  postprocessor, priorbox
from vizer.draw import draw_boxes
from PIL import Image
from Data.Transfroms import SSDTramsfrom
import numpy as np
import time

__all__ = ['SSD']

class SSD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
       
        self.backbone = vgg(cfg, pretrained=True)   
       
        self.predictor = predictor(cfg)
        self.postprocessor = postprocessor(cfg)
        self.priors = priorbox(self.cfg)()

    def forward(self, images):
       
        features = self.backbone(images)
       
        cls_logits, bbox_pred = self.predictor(features)
        return cls_logits, bbox_pred

    def load_pretrained_weight(self, weight_pkl):
        self.load_state_dict(torch.load(weight_pkl))

    def forward_with_postprocess(self, images):
       
        cls_logits, bbox_pred = self.forward(images)
        detections = self.postprocessor(cls_logits, bbox_pred)
        return detections

    @torch.no_grad()
    def Detect_single_img(self, image, score_threshold=0.7, device='cuda'):
        
        self.eval()
        assert isinstance(image,Image.Image)
        w, h = image.width, image.height
        images_tensor = SSDTramsfrom(self.cfg, is_train=False)(np.array(image))[0].unsqueeze(0)

        self.to(device)
        images_tensor = images_tensor.to(device)
        time1 = time.time()
        detections = self.forward_with_postprocess(images_tensor)[0]
        boxes, labels, scores = detections
        boxes, labels, scores = boxes.to('cpu').numpy(), labels.to('cpu').numpy(), scores.to('cpu').numpy()
        boxes[:, 0::2] *= (w / self.cfg.MODEL.INPUT.IMAGE_SIZE)
        boxes[:, 1::2] *= (h / self.cfg.MODEL.INPUT.IMAGE_SIZE)

        indices = scores > score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        print("Detect {} object, inference cost {:.2f} ms".format(len(scores),(time.time()-time1)*1000))
       
        
        return image, boxes, labels, scores

    @torch.no_grad()
    def Detect_video(self, video_path, score_threshold=0.5, save_video_path=None, show=True):
        
        import cv2
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        weight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if save_video_path:
            out = cv2.VideoWriter(save_video_path, fourcc, cap.get(5), (weight, height))
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                drawn_image, boxes, labels, scores =self.Detect_single_img(image=image,
                                                                           device='cuda:0',
                                                                           score_threshold=score_threshold)
                frame = cv2.cvtColor(np.asarray(drawn_image), cv2.COLOR_RGB2BGR)
                if show:
                    cv2.imshow('frame', frame)
                if save_video_path:
                    out.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        if save_video_path:
            out.release()
        cv2.destroyAllWindows()
        return True
