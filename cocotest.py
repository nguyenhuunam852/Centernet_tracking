# %%
from pycocotools.coco import COCO
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
val_annot_path = 'faces4coco-2017-coco-annotations\cocoface\cocoface_instances_val2017.json'
coco = COCO(val_annot_path)

imgIds = coco.getImgIds()
imgIds.sort()
imgid = imgIds[4]
img_info = coco.loadImgs(imgid)[0]

annIds = coco.getAnnIds(imgIds=[imgid])
anns = coco.loadAnns(annIds)

fig, ax = plt.subplots()

original = cv2.imread(
    r'D:\train2017\val2017\{}'.format(img_info['file_name']))
ax.imshow(original)
print(len(anns))
for annotation in anns:
    if len(annotation['bbox']) > 0:
        bbox = annotation['bbox']
        rect = patches.Rectangle(
            (int(bbox[0]), int(bbox[1])), int(bbox[2]), int(bbox[3]), linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

plt.show()
# %%
