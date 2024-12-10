import os
import cv2
import torch
import imutils
import numpy as np
from diffusers.utils import load_image
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
depth_estimator = DPTForDepthEstimation.from_pretrained("checkpoints/Intel_dpt-hybrid-midas").to("cuda")
feature_extractor = DPTFeatureExtractor.from_pretrained("checkpoints/Intel_dpt-hybrid-midas")


def get_depth_map(image):
    h, w, _ = np.asarray(image).shape
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth
    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)
    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image


def depth_condition(path):
    # input: .tiff
    # depth_img = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    # depth_img = np.tile(depth_img[..., None], (1, 1, 3))
    # depth_img = (depth_img - np.min(depth_img)) / (np.max(depth_img) - np.min(depth_img))
    # depth_img = (1 - depth_img) * 255
    # depth_img = (65535 - depth_img) / 65535 * 255
    depth_img = cv2.imread(path).astype(np.float32)
    # depth_img = (depth_img - np.min(depth_img)) / (np.max(depth_img) - np.min(depth_img))
    image = Image.fromarray(depth_img.astype(np.uint8)) # [h, w, 3]
    return image


def get_canny_map(img):
    image = np.asarray(load_image(img))
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)


def refine_mask_18(mask, direction):
    h, w, _ = mask.shape
    if direction == "right":
        for i in range(h):
            temp = np.nonzero(1-mask[i])[0]
            if len(temp) > 0:
                mask[i, :temp[-1]] = 0
    elif direction == "left":
        for i in range(h):
            temp = np.nonzero(1-mask[i])[0]
            if len(temp) > 0:
                mask[i, temp[0]:] = 0
    elif direction == "up":
        mask[:810] = 1
        mask[810:, 30:1225] = 0
        for i in range(30):
            temp = np.nonzero(1-mask[:, i])[0]
            if len(temp) > 0:
                mask[temp[0]:, i] = 0
        for i in range(1225, w):
            temp = np.nonzero(1-mask[:, i])[0]
            if len(temp) > 0:
                mask[temp[0]:, i] = 0
    elif direction == "down":
        mask[250:] = 1
        mask[:250, 60:1245] = 0
        for i in range(60):
            temp = np.nonzero(1-mask[:, i])[0]
            if len(temp) > 0:
                mask[:temp[-1], i] = 0
        for i in range(1245, w):
            temp = np.nonzero(1-mask[:, i])[0]
            if len(temp) > 0:
                mask[:temp[-1], i] = 0
    else:
        raise ValueError("ERROR IN DIRECTION!")
    return mask



def refine_mask(mask, direction):
    # Image.fromarray((np.tile(mask, (1, 1, 3))*255).astype(np.uint8)).save(os.path.join(direction+'_mask.jpg'))
    if direction == "right":
        w = np.nonzero(np.min(mask, 0))[0][0]
        h1 = np.nonzero(1-mask[:, w-5])[0][0]
        h2 = np.nonzero(1-mask[:, w-5])[0][-1]
        mask[h1:h2, :w] = 0
        mask[:, w-5:] = 1
    elif direction == "left":
        w = np.nonzero(np.min(mask, 0))[0][-1]
        h1 = np.nonzero(1-mask[:, w+5])[0][0]
        h2 = np.nonzero(1-mask[:, w+5])[0][-1]
        mask[h1:h2, w:] = 0
        mask[:, :w+5] = 1
    elif direction == "up":
        h = np.nonzero(1-np.min(mask, 1))[0][0]
        w1 = np.nonzero(1-mask[h+5])[0][0]
        w2 = np.nonzero(1-mask[h+5])[0][-1]
        mask[h:, w1:w2] = 0
        mask[:h+5] = 1
    elif direction == "down":
        h = np.nonzero(1-np.min(mask, 1))[0][-1]
        w1 = np.nonzero(1-mask[h-5])[0][0]
        w2 = np.nonzero(1-mask[h-5])[0][-1]
        mask[:h, w1:w2] = 0
        mask[h-5:] = 1
    else:
         raise NameError('ERROR IN DIRECTION')

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(7, 7))
    eroded_mask = cv2.erode(mask[:, :, 0], kernel)
    dilated_mask = cv2.dilate(eroded_mask, kernel)
    refined_mask = dilated_mask[..., None]
    # Image.fromarray((np.tile(dilated_mask[..., None], (1, 1, 3))*255).astype(np.uint8)).save(os.path.join('mask_2.jpg'))
    return refined_mask


def generate_inpainting_datasets():
    INPAINT_DIR = "datasets/diffusion_inpainting"
    with open(TRAJECTORY_PATH, "r") as f:
        trajectory_list = json.load(f)
        f.close()
    with open(SCANVP_PATH, "r") as f:
        scanvp_list = json.load(f)
        f.close()
    i = -1
    for trajectory in trajectory_list:
        for j in range(len(trajectory["path"])-1):
            i += 1
            cv2.imread(os.path.join(INPAINT_DIR, "data", "mask_"+str(i)))
            scan, vp, idx, _ = re.split("/|\.", trajectory["path"][j])
            _, next_vp, _, _ = re.split("/|\.", trajectory["path"][j+1])
            guidance, mask = guide(
                os.path.join(RGB_DIR, trajectory["path"][j]),
                scanvp_list[scan+"_"+vp][next_vp][1:]
            )
            Image.fromarray((guidance*255).astype(np.uint8)).resize((512, 512)).save(os.path.join(INPAINT_DIR, "data", "guidance_"+str(i)+".jpg"))
            Image.fromarray((255*np.tile(mask, (1, 1, 3)).astype(np.uint8))).resize((512, 512)).save(os.path.join(INPAINT_DIR, "data", "mask_"+str(i)+".jpg"))
            load_image(os.path.join(RGB_DIR, scan, next_vp, idx+".jpg")).resize((512, 512)).save(os.path.join(INPAINT_DIR, "data", "image_"+str(i)+".jpg"))         
    with open(os.path.join(INPAINT_DIR, "data.csv"), "w") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "mask_path", "guidance_path", "partition"])
        for j in range(i):
            writer.writerow(["image_"+str(j)+".jpg", "mask_"+str(j)+".jpg", "guidance_"+str(j)+".jpg", "train"])
        writer.writerow(["image_0.jpg", "mask_0.jpg", "guidance_0.jpg", "validation"])
