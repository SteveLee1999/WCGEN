import re
import os
import json
import torch
import argparse
import numpy as np
import tensorflow as tf
from scipy.spatial.transform import Rotation as R
from diffusers.utils import load_image
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, StableDiffusionXLControlNetPipeline, AutoPipelineForInpainting, StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from utils import *


DEVICE = "cuda"
INVALID_RGB_VALUE = -1 # Negative value to avoid collision with black pixels.
DEPTH_SCALE = 20.0
EQUI_HEIGHT = 1920 # for equirectangular
PERS_HEIGHT, PERS_WIDTH = 960, 1280 # for perspective image
SCALING = 5.0 # scaling factor for depth map
BIAS = 0.0 # bias factor for depth map
CAMERA_EXTRINSICS = np.array(  # should be replaced with Matterport3D data
    [[0.99992853, -0.01185191, 0.00158339],
     [0.01187317, 0.9998291, -0.01416931],
     [-0.00141519, 0.01418709, 0.9998984 ]], np.float32)
CAMERA_INTRINSICS = np.array(  # should be replaced with Matterport3D data
    [[825., 0., 630.],
     [0., 825., 505.],
     [0., 0., 1.]], np.float32)
BATCH_SIZE = 1
GENERATOR = torch.Generator(device=DEVICE).manual_seed(324)
PIPE_1 = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,\
    controlnet=ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16
    ),
)
PIPE_1.to(DEVICE)
PIPE_2 = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,\
    controlnet=ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16
    ),
)
PIPE_2.to(DEVICE)
PIPE_3 = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16,\
    controlnet=ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16
    ),
)
PIPE_3.to(DEVICE)
MASK_PROCESSOR = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16
).to(DEVICE).mask_processor
CAPTION_PATH = "results/captions_blip2_36_R2R_train.json"
TRAJECTORY_PATH = "results/list_train.json"
SCANVP_PATH = "datasets/R2R/annotations/scanvp_candview_relangles.json"
RGB_DIR = "datasets/Matterport3D/rgb"
OUTPUT_DIR = "test"


def guide(rgb_path, tvec=[0., 0., 0.], angle=[0., 0., 0.], debug=False):
    with tf.io.gfile.GFile(rgb_path, 'rb') as f:
        input_rgb_frames = tf.image.decode_jpeg(f.read())
        input_rgb_frames = tf.image.resize(input_rgb_frames, (PERS_HEIGHT, PERS_WIDTH), method='bilinear')
        input_rgb_frames = tf.cast(input_rgb_frames, tf.uint8)
        # (960, 1280, 3)
    input_depth_frames = get_depth_map(load_image(rgb_path).resize((PERS_WIDTH, PERS_HEIGHT)))
    input_depth_frames = tf.convert_to_tensor(np.asarray(input_depth_frames), tf.float32)[..., 0:1]
    midas_max = tf.reduce_max(input_depth_frames)
    input_depth_frames = midas_max - input_depth_frames
    input_depth_frames = input_depth_frames / midas_max / SCALING + BIAS
    input_depth_frames = tf.clip_by_value(input_depth_frames, 0, 1)
    # MiDaS (960, 1280, 1)

    rgb_tensor = project_perspective_image(
        image=tf.image.convert_image_dtype(input_rgb_frames, tf.float32),
        fov=None,
        output_height=EQUI_HEIGHT,
        camera_intrinsics=CAMERA_INTRINSICS,
        camera_extrinsics=CAMERA_EXTRINSICS,
        round_to_nearest=True)
    depth_tensor = project_perspective_image(
        image=tf.image.convert_image_dtype(input_depth_frames, tf.float32),
        fov=None,
        output_height=EQUI_HEIGHT,
        camera_intrinsics=CAMERA_INTRINSICS,
        camera_extrinsics=CAMERA_EXTRINSICS,
        round_to_nearest=True)

    proj_rgb_tensor = tf.cast(rgb_tensor[None, ...] * 255, tf.int32)
    proj_depth_tensor = depth_tensor[None, ..., 0]
    
    # Add points to point cloud memory.
    xyz1, feats = equirectangular_to_pointcloud(
        proj_rgb_tensor, proj_depth_tensor,
        INVALID_RGB_VALUE, DEPTH_SCALE)

    # Example translation vector and rotation.
    # translation_in_meters = 0.2 #@param {type:"number"}
    # tvec = [0, 0, translation_in_meters / DEPTH_SCALE]
    # [right(+), down(+), foward(+)]

    new_rotation = R.from_rotvec([i / 180 * np.pi for i in angle]).as_matrix()
    new_rot_mat = CAMERA_EXTRINSICS @ new_rotation

    relative_position = tf.stack(
        [tvec[0], tvec[2], -tvec[1], tf.zeros(())], axis=-1)[None, :]
    relative_coords = xyz1 - relative_position[..., None]

    # project point-cloud to equirectangular.
    pred_depth, pred_rgb = (
        project_feats_to_equirectangular(
            feats, relative_coords, EQUI_HEIGHT, EQUI_HEIGHT * 2,
            INVALID_RGB_VALUE, DEPTH_SCALE))

    # generate mask
    pred_mask = tf.cast(
            tf.math.logical_and(
            tf.math.logical_and(pred_depth != 1.0, pred_depth != 0.0),
            tf.math.reduce_all(pred_rgb != 0.0, axis=-1)), tf.float32)
    pred_mask = get_perspective_from_equirectangular_image(
        pred_mask[0][..., None], CAMERA_INTRINSICS, new_rot_mat, PERS_HEIGHT, PERS_WIDTH)
    pred_mask = tf.cast(pred_mask[None, ...] == 1.0, tf.float32)
    pred_mask = 1 - pred_mask  # shape=[1, h, w, 1], value 1 means masked

    # convert into perspective images here:
    pers_rgb_guidance = get_perspective_from_equirectangular_image(
        pred_rgb[0], CAMERA_INTRINSICS, new_rot_mat, PERS_HEIGHT, PERS_WIDTH)
    pers_rgb_guidance = tf.clip_by_value(pers_rgb_guidance / 255, 0, 1)

    # pers_depth_guidance = get_perspective_from_equirectangular_image(
    #     pred_depth[0][..., None], CAMERA_INTRINSICS, new_rot_mat, PERS_HEIGHT, PERS_WIDTH)
    # inputs = {
    #     'proj_image': (1-pred_mask) * pers_rgb_guidance[None, ...],
    #     'proj_depth': (1-pred_mask) * pers_depth_guidance[None, ...],
    #     'proj_mask': (1-pred_mask),
    #     'blurred_mask': tf.zeros((1, PERS_HEIGHT, PERS_WIDTH, 1)),
    # }
    # (_, _, _, depth_out, _, _, rgb_out) = SE3D_MODEL.model([inputs, None], training=False)
    # return np.asarray(pers_rgb_guidance).copy(), rgb_out

    return np.asarray(pers_rgb_guidance).copy(), np.asarray(pred_mask[0]).copy()


def f1(idx):
    if int(idx) < 12:
        return str(int(idx)+12)
    elif int(idx) > 23:
        return str(int(idx)-12)
    else:
        return idx


def f2(a, b):
    gap = (b - a) % 12
    return gap - 12 if gap > 6 else gap   


def trajectory_stage(index, gap=4675):
    with open(TRAJECTORY_PATH, "r") as f:
        trajectory_list = json.load(f)
        f.close()
    with open(SCANVP_PATH, "r") as f:
        scanvp_list = json.load(f)
        f.close()
    with open(CAPTION_PATH, "r") as f:
        captions = json.load(f)
        f.close()
    for trajectory in trajectory_list[index*gap:(index+1)*gap]:
        for i in range(len(trajectory["path"])-1):
            scan, vp, idx, _ = re.split("/|\.", trajectory["path"][i])
            next_scan, next_vp, next_idx, _ = re.split("/|\.", trajectory["path"][i+1])
            idx = f1(idx)
            next_idx = f1(next_idx)
            if i == 0:
                prompts = [captions[scan+"_"+vp+"_"+idx]]
                images = [load_image(os.path.join(RGB_DIR, scan, vp, idx+".jpg")).resize((PERS_WIDTH, PERS_HEIGHT))]
                conditions = [get_depth_map(images[0])]
                outputs = PIPE_1(prompt=prompts, image=conditions, num_inference_steps=250, \
                                 guidance_scale=12).images
                os.makedirs(os.path.join(OUTPUT_DIR, str(trajectory["instruction_id"]), scan, vp), exist_ok=True)
                outputs[0].resize((PERS_WIDTH, PERS_HEIGHT)).save(os.path.join(OUTPUT_DIR, str(trajectory["instruction_id"]), scan, vp, idx+".jpg"))
            rgb = load_image(os.path.join(RGB_DIR, next_scan, next_vp, next_idx+".jpg")).resize((PERS_WIDTH, PERS_HEIGHT))
            rotation_gap = f2(int(idx), int(next_idx))
            if abs(rotation_gap) > 1:
                image = rgb
            else:
                tvec = scanvp_list[scan+"_"+vp][next_vp][1:]
                angle = [0, -30*rotation_gap, 0]
                guidance, mask = guide( \
                    os.path.join(OUTPUT_DIR, str(trajectory["instruction_id"]), scan, vp, idx+".jpg"), \
                    [tvec[1], -tvec[2], 3.5*tvec[0]], \
                    angle)
                guidance = np.where((mask == 0), guidance, np.asarray(rgb)/255.)
                image = Image.fromarray((255*guidance).astype(np.uint8))
            prompts = [captions[next_scan+"_"+next_vp+"_"+next_idx]]
            images = [image]
            conditions = [get_depth_map(rgb)]
            outputs = PIPE_2(prompt=prompts, image=images, control_image=conditions, num_inference_steps=250, \
                            strength=0.95, guidance_scale=12, generator=GENERATOR).images              
            os.makedirs(os.path.join(OUTPUT_DIR, str(trajectory["instruction_id"]), next_scan, next_vp), exist_ok=True)
            outputs[0].resize((PERS_WIDTH, PERS_HEIGHT)).save(os.path.join(OUTPUT_DIR, str(trajectory["instruction_id"]), next_scan, next_vp, next_idx+".jpg"))
            exit()


def viewpoint_stage(index, gap=4675):
    with open(TRAJECTORY_PATH, "r") as f:
        trajectory_list = json.load(f)
        f.close()
    with open(CAPTION_PATH, "r") as f:
        captions = json.load(f)
        f.close()
    for trajectory in trajectory_list[index*gap:(index+1)*gap]:
        for i in range(len(trajectory["path"])):
            scan, vp, start_idx, _ = re.split("/|\.", trajectory["path"][i])   
            start_idx = int(f1(start_idx))
            save_dir = os.path.join(OUTPUT_DIR, str(trajectory["instruction_id"]), scan, vp)
            for j in range(start_idx, start_idx+12):
                targets = []
                prompts = []
                images = []
                masks = []
                conditions = []
                idx = (j % 12) + 12
                for d in ["right", "up", "down"]:
                    if d == "right":
                        target_idx = (idx + 1) % 12 + 12
                        rgb = cv2.cvtColor(cv2.imread(os.path.join(RGB_DIR, scan, vp, str(target_idx)+".jpg")), cv2.COLOR_BGR2RGB)
                        rgb = np.asarray(Image.fromarray(rgb).resize((1280, 960))).astype(np.float32) / 255.
                        guidance, _ = guide(
                            rgb_path=os.path.join(save_dir, str(idx)+".jpg"),
                            angle = [0, -30, 0]
                        )
                        mask = np.asarray(MASK_PROCESSOR.blur(load_image("mask/right.jpg"), blur_factor=33)) / 255.
                        if target_idx == start_idx:
                            continue
                        elif (target_idx + 1) % 12 + 12 != start_idx:
                            guidance = (1-mask)*guidance + mask*rgb
                        else:
                            guidance_left, _ = guide(
                                rgb_path=os.path.join(save_dir, str(start_idx)+".jpg"),
                                angle = [0, 30, 0]
                            )
                            mask_left = np.asarray(MASK_PROCESSOR.blur(load_image("mask/left.jpg"), blur_factor=33)) / 255.
                            guidance = mask_left*guidance + (1-mask_left)*guidance_left
                            mask = mask * mask_left
                            guidance = (1-mask)*guidance + mask*rgb
                    elif d == "up":
                        target_idx = idx + 12
                        rgb = cv2.cvtColor(cv2.imread(os.path.join(RGB_DIR, scan, vp, str(target_idx)+".jpg")), cv2.COLOR_BGR2RGB)
                        rgb = np.asarray(Image.fromarray(rgb).resize((1280, 960))).astype(np.float32) / 255.
                        guidance, _ = guide(
                            rgb_path=os.path.join(save_dir, str(idx)+".jpg"),
                            angle=[-30, 0, 0]
                        )
                        if idx == start_idx:
                            mask = np.asarray(MASK_PROCESSOR.blur(load_image("mask/start_up.jpg"), blur_factor=33)) / 255.
                            guidance = (1-mask)*guidance + mask*rgb
                        elif (idx + 1) % 12 + 12 != start_idx:
                            mask = np.asarray(MASK_PROCESSOR.blur(load_image("mask/up.jpg"), blur_factor=33)) / 255.
                            guidance_right, _ = guide(
                                rgb_path=os.path.join(save_dir, str((idx-1)%12+24)+".jpg"),
                                angle=[0, -26, 15.25]
                            )
                            mask_right = np.asarray(MASK_PROCESSOR.blur(load_image("mask/up_right.jpg"), blur_factor=33)) / 255.
                            guidance = mask_right*guidance + (1-mask_right)*guidance_right
                            mask = mask * mask_right
                            guidance = (1-mask)*guidance + mask*rgb
                        else:
                            mask = np.asarray(MASK_PROCESSOR.blur(load_image("mask/up.jpg"), blur_factor=33)) / 255.
                            guidance_right, _ = guide(
                                rgb_path=os.path.join(save_dir, str((idx-1)%12+24)+".jpg"),
                                angle=[0, -26, 15.25]
                            )
                            mask_right = np.asarray(MASK_PROCESSOR.blur(load_image("mask/up_right.jpg"), blur_factor=33)) / 255.
                            guidance_left, _ = guide(
                                rgb_path=os.path.join(save_dir, str(start_idx+12)+".jpg"),
                                angle=[0, 26, -15.25]
                            )
                            mask_left = np.asarray(MASK_PROCESSOR.blur(load_image("mask/up_left.jpg"), blur_factor=33)) / 255.
                            guidance = mask_left*guidance + (1-mask_left)*guidance_left
                            guidance = mask_right*guidance + (1-mask_right)*guidance_right
                            mask = mask * mask_left * mask_right
                            guidance = (1-mask)*guidance + mask*rgb
                    elif d == "down":  
                        target_idx = idx - 12
                        rgb = cv2.cvtColor(cv2.imread(os.path.join(RGB_DIR, scan, vp, str(target_idx)+".jpg")), cv2.COLOR_BGR2RGB)
                        rgb = np.asarray(Image.fromarray(rgb).resize((1280, 960))).astype(np.float32) / 255.
                        guidance, _ = guide(
                            rgb_path=os.path.join(save_dir, str(idx)+".jpg"),
                            angle = [30, 0, 0]
                        )
                        if idx == start_idx:
                            mask = np.asarray(MASK_PROCESSOR.blur(load_image("mask/start_down.jpg"), blur_factor=33)) / 255.
                            guidance = (1-mask)*guidance + mask*rgb
                        elif (idx + 1) % 12 + 12 != start_idx:
                            mask = np.asarray(MASK_PROCESSOR.blur(load_image("mask/down.jpg"), blur_factor=33)) / 255.
                            guidance_right, mask_right = guide(
                                rgb_path=os.path.join(save_dir, str((idx-1)%12)+".jpg"),
                                angle=[0, -26, -14.25]
                            )
                            mask_right = np.asarray(MASK_PROCESSOR.blur(load_image("mask/down_right.jpg"), blur_factor=33)) / 255.
                            guidance = mask_right*guidance + (1-mask_right)*guidance_right
                            mask = mask * mask_right
                            guidance = (1-mask)*guidance + mask*rgb
                        else:
                            mask = np.asarray(MASK_PROCESSOR.blur(load_image("mask/down.jpg"), blur_factor=33)) / 255.
                            guidance_right, _ = guide(
                                rgb_path=os.path.join(save_dir, str((idx-1)%12)+".jpg"),
                                angle=[0, -26, -14.25]
                            )
                            mask_right = np.asarray(MASK_PROCESSOR.blur(load_image("mask/down_right.jpg"), blur_factor=33)) / 255.
                            guidance_left, _ = guide(
                                rgb_path=os.path.join(save_dir, str(start_idx-12)+".jpg"),
                                angle=[0, 26, 16.25]
                            )
                            mask_left = np.asarray(MASK_PROCESSOR.blur(load_image("mask/down_left.jpg"), blur_factor=33)) / 255.
                            guidance = mask_left*guidance + (1-mask_left)*guidance_left
                            guidance = mask_right*guidance + (1-mask_right)*guidance_right
                            mask = mask * mask_left * mask_right
                            guidance = (1-mask)*guidance + mask*rgb                    
                    targets.append(target_idx)
                    prompts.append("photorealistic, 32K, extremely detailed, "+captions[scan + "_" + vp + "_" + str(target_idx)])
                    images.append(Image.fromarray((255*guidance).astype(np.uint8)))
                    masks.append(Image.fromarray((255*mask).astype(np.uint8)))
                    conditions.append(get_depth_map(images[-1]))
                    # Image.fromarray((np.asarray(guidance)*255).astype(np.uint8)).save(d+"_guidance.jpg")
                    # Image.fromarray((mask*255).astype(np.uint8)).save(os.path.join(d+"_mask.jpg"))
                outputs = PIPE_3(prompt=prompts, image=images, mask_image=masks, control_image=conditions, generator=GENERATOR, \
                                 guidance_scale=12, controlnet_conditioning_scale=0.25, \
                                 negative_prompt="deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW", \
                                 num_inference_steps=250).images
                for k in range(len(targets)):
                    outputs[k].resize((PERS_WIDTH, PERS_HEIGHT)).save(os.path.join(save_dir, str(targets[k]) + ".jpg"))
                torch.cuda.empty_cache()


def preprocess(base_dir, mode):
    if mode == "trajectory":
        with open(TRAJECTORY_PATH, "r") as f:
            trajectory_list = json.load(f)
            f.close()
        todo_list = []
        for trajectory in trajectory_list:
            todo = False
            for i in range(len(trajectory["path"])):
                scan, vp, _, _ = re.split("/|\.", trajectory["path"][i])
                path = os.path.join(base_dir, str(trajectory["instruction_id"]), scan, vp)
                if not os.path.exists(path):
                    todo = True
                    break
                elif len(os.listdir(path)) == 0:
                    todo = True
                    break
            if todo:
                todo_list.append(trajectory)
        print(len(todo_list))
        with open('results/list_train_temp.json', 'w') as json_file:
            json.dump(todo_list, json_file, indent=4)
    elif mode == "viewpoint":
        with open(TRAJECTORY_PATH, "r") as f:
            trajectory_list = json.load(f)
            f.close()
        todo_list = []
        for trajectory in trajectory_list:
            todo = False
            for i in range(len(trajectory["path"])):
                scan, vp, _, _ = re.split("/|\.", trajectory["path"][i])
                path = os.path.join(base_dir, str(trajectory["instruction_id"]), scan, vp)
                if len(os.listdir(path)) < 36:
                    todo = True
                    break
            if todo:
                todo_list.append(trajectory)
        print(len(todo_list))
        with open('results/list_train_temp.json', 'w') as json_file:
            json.dump(todo_list, json_file, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="index for task")
    args = parser.parse_args()
    # trajectory_stage(args.index)
    viewpoint_stage(args.index)
    # preprocess("results/wcgen_2_trajectory", "trajectory")
