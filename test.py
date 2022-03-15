from imgaug import augmenters as iaa
from drive.MyDrive.Mask_RCNN.mrcnn.config import Config
from drive.MyDrive.Mask_RCNN.mrcnn import model as modellib
from drive.MyDrive.Mask_RCNN.mrcnn import visualize
from drive.MyDrive.Mask_RCNN.mrcnn import utils
from imutils import paths
import numpy as np
import argparse
import imutils
import random
import cv2
import os
import json
from google.colab.patches import cv2_imshow

dataset_path = os.path.abspath("/content/drive/MyDrive/trashnet")
images_path = os.path.abspath("/content/drive/MyDrive/trashnet/dataset")
masks_path = os.path.sep.join([dataset_path, "masks_modif.json"])



training_split = 0.75

image_paths = sorted(list(paths.list_images(images_path)))
idxs = list(range(0, len(image_paths) ))
random.seed(42)
random.shuffle(idxs)
i = int(len(idxs) * training_split)
train_idxs = idxs[:i]
val_idxs = idxs[i:]


#class_names = {1 : "cardboard", 2 : "glass", 3: "paper", 4: "plastic", 5: "metal", 6: "trash"}
class_names ={1: "trash"}

coco_path = "/content/drive/MyDrive/mask_rcnn_coco.h5"

LOGS_AND_MODEL_DIR = "logs"


class TrashConfig(Config):
  NAME = 'trash'
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1
  STEPS_PER_EPOCH = len(train_idxs) // (IMAGES_PER_GPU * GPU_COUNT)
  NUM_CLASSES = len(class_names) + 1


class TrashInferenceConfig(TrashConfig):
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1
  DETECTION_MIN_CONFIDENCE = 0.9


class TrashDataset(utils.Dataset):
  def __init__(self, imagePaths, masksPath, classNames, width = 1024):
    super().__init__(self)
    self.imagePaths = imagePaths
    self.classNames = classNames
    self.width = width
    self.annots = self.load_mask_data(masksPath)

  def load_annotation_data(self, masksPath):
   annotations = json.loads(open(masksPath).read())
   annots = {}

   for(fileID, data) in sorted(open(masksPath).read()):
     annots[data["filename"]] = data

   return annots

  def load_trash(self, idxs):
    for (class_id, label) in  self.classNames.items():
      self.add_class("trash", class_id, label)

    
    image_path = os.listdir(self.imagePaths)
    for i in idxs:
      
      #image_path = self.imagePaths[i]
      #filename = image_path[i].split(os.path.sep)[-1]
      
      
      k = self.imagePaths + '/' + i
      filename = k.split(os.path.sep)[-1]
      image = cv2.imread(k)
      (origH, origW) = image.shape[:2]
      image = imutils.resize(image, width = self.width)
      (newH, newW) = image.shape[:2]
      self.add_image("trash", image_id = filename, width = newW, height = newH,
                     orig_width = origW, orig_height=origH,
                     path = k)
  
  def load_image(self, image_id):
   p = self.image_info[image_id]["path"]
   image = cv2.imread(p)
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   image = imutils.resize(image, width = self.width)
   return image

  def load_mask_data(self, image_id):
   info = self.image_info[image_id]
   annot = self.annots[info["id"]]

   masks = np.zeros((info["height"], info["width"], len(annot["regions"])), dtype = "uint8")

   for (i, region) in enumerate(annot["regions"]) :
    region_mask = np.zeros(masks.shape[:2], dtype = "uint8")
    sa = region["shape_attributes"]
    ra = region["region_attributes"]
    ratio = info["width"] / float(info["orig_width"])
    cX = int(sa["cx"] * ratio)
    cY = int(sa["cy"] * ratio)
    r = int(sa["r"] * ratio)

    

    cv2.circle(region_mask, (cX, cY), r, 1, -1)
    
    masks[:, :, i] = region_mask

   return (masks.astype("bool"), np.ones((masks.shape[-1], ), dtype ="int32"))


train_dataset = TrashDataset(images_path, masks_path, class_names)
train_dataset.load_trash(train_idxs)
train_dataset.prepare()


val_dataset = TrashDataset(images_path, masks_path, class_names)
val_dataset.load_trash(val_idxs)
val_dataset.prepare()


config = TrashConfig()
config.display()


model = modellib.MaskRCNN(mode = "training", config = config, model_dir = LOGS_AND_MODEL_DIR)
model.load_weights(coco_path, by_name = True, exclude = ["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

model.train(train_dataset, val_dataset, epochs = 20, layers = "heads", learning_rate = config.LEARNING_RATE)
model.train(train_dataset, val_dataset, epochs = 40, layers = "all", learning_rate = config.LEARNING_RATE / 10)


config = TrashInferenceConfig()
model = modellib.MaskRCNN(mode = "inference", config = config, model_dir = logs_and_model_dir)

weights = model.find_last()
model.load_weigths(weights, by_name = True )


image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = imutils.resize(image, width = 1024)

r = model.detect([image], verbose = 1)[0]

for i in range(0, r["rois"].shape[0]):
  mask = r["masks"][:, :, i]
  image = visualize.apply_mask(image, mask, (1.0, 0.0, 0.0), alpha = 0.5)
  image = visualize.draw_box(image, r["rois"][i], [1.0, 0.0, 0.0])
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  for i in range(0, len(r["scores"])):
   (startY, startX, endY, endX) = r["rois"][i]
   class_id = r["class_ids"][i]
   label = class_names[class_id]
   score = r["scores"][i]

   text = "{}: {:.4f}".format(label, score)
   y = startY - 10 if startY - 10 > 10 else startY + 10
   cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
      
image = imutils.resize(image, width = 512)
cv2.imshow("Output", image)
cv2.waitKey(0)