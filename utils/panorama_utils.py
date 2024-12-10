import os
import cv2
import numpy as np
import imutils
import open3d as o3d


def visualize_obj(path):
    mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
    print(np.asarray(mesh.vertices).shape)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])
    

def stitch_images(img_path, output_path):
    """
    images: a list of path
    """
    imgs = []
    for path in img_path:
        imgs.append(cv2.imread(path))
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(imgs)
    if status == 0: # status==0 means success
        cv2.imwrite(output_path, stitched)
    else:
        print("error")

