
from Model import SSD
from Configs import _C as cfg
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image,ImageFont, ImageDraw

net = SSD(cfg)

net.to('cuda')
# loading weight
net.load_pretrained_weight('Weights/pretrained/vgg_ssd300_voc0712.pkl')
# open image
image = Image.open("Images/004347.jpg")
# detection
drawn_image, boxes, labels, scores = net.Detect_single_img(image=image,score_threshold=0.5)
print(labels)
plt.imsave('Images/000531_det.jpg',drawn_image)
print(boxes.shape)

for i, box in enumerate(boxes):
  
            
            
            predicted_class = labels[i]
            score = scores[i]

            top, left, bottom, right = boxes[i]
         

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))
          
            
            class_name= ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

            label = '{} {:.2f}'.format(class_name[predicted_class], score)
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype('arial.ttf', 24)
            label_size = draw.textsize(label)
            label = label.encode('utf-8')
            print(label)
            
            
            draw = ImageDraw.Draw(image)
            
            
           
            draw.ellipse(
                    [top , left , bottom, right ],outline=255)
            draw.text([top,left],str(label,'UTF-8'),(255,0,0),font=font)
            print(labels)
            del draw
plt.imshow(image)
#plt.imshow(drawn_image)
plt.show()