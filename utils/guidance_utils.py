"""Utils function for panorama processing."""
import math
from typing import Optional, Tuple
import numpy as np
import tensorflow as tf
from tensorflow_addons import image as tfa_image
import numpy as np


def get_world_to_image_transform(image_shape,
                                 fov,
                                 camera_intrinsics: Optional[tf.Tensor] = None,
                                 rotations: Optional[Tuple[int, int]] = None,
                                 rotation_matrix: Optional[tf.Tensor] = None):
  """Returns a 3x3 transformation matrix from world to image coordinates.

  The image is oriented orthogonally to the x axis. The x axis of the image
  points away from the z axis in world coordinates, and the y axis of the image
  points away from the y axis in world coordinates.

  Modified from //experimental/earthsea/wanderer/geometry_utils.py.

  Args:
    image_shape: list with shape of the image (height, width).
    fov: tensor with the fields of view of the image (vertical, horizontal) in
      radians.
    camera_intrinsics: Optional camera intrinsics matrix to use instead of
      computing it using field of view.
    rotations: optional tensor containing pitch and heading in radians for
      rotating camera. A positive pitch rotates the camera upwards. A positive
      heading rotates the camera along the equator clockwise.
    rotation_matrix: Optional 3x3 rotation matrix to use instead as an
      alternative to the rotations parameter.

  Returns:
    A 3x3 tensor that transforms world to image coordinates.
  """
  if camera_intrinsics is None:
    height, width = image_shape
    fov_y, fov_x = tf.unstack(fov)

    fx = 0.5 * (width - 1.0) / tf.tan(fov_x / 2)
    fy = 0.5 * (height - 1.0) / tf.tan(fov_y / 2)

    camera_intrinsics = tf.stack([
        tf.stack([fx, 0, 0.5 * (width - 1)]),
        tf.stack([0, fy, 0.5 * (height - 1)]),
        tf.stack([0., 0, 1])
    ])
  if rotations is not None:
    rot_pitch, rot_heading = tf.unstack(rotations)
    pitch_rotation = tf.stack([
        tf.stack([1., 0, 0]),
        tf.stack([0, tf.cos(-rot_pitch), -tf.sin(-rot_pitch)]),
        tf.stack([0, tf.sin(-rot_pitch), tf.cos(-rot_pitch)])
    ])
    heading_rotation = tf.stack([
        tf.stack([tf.cos(-rot_heading), 0, tf.sin(-rot_heading)]),
        tf.stack([0., 1, 0]),
        tf.stack([-tf.sin(-rot_heading), 0, tf.cos(-rot_heading)])
    ])
    extrinsics = pitch_rotation @ heading_rotation
  elif rotation_matrix is not None:
    extrinsics = rotation_matrix
  else:
    extrinsics = tf.constant([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
    ])
  transform = camera_intrinsics @ extrinsics
  return transform


def equirectangular_pixel_rays(output_height):
  """Generates a 3d point on a unit ball for each equirectangular image pixel.


  The output coordinate system is x-right, y-down, z-forward at the center of
  the equirectangular image.

  Args:
    output_height: Int height of the equirectangular image.

  Returns:
    pixel_rays: an [3, output_height * output_width] tensor containing
      an xyz coordinate on the unit-radius ball for each pixel.
  """
  output_width = tf.cast(tf.cast(output_height, tf.float32) * 2, tf.int32)
  heading = tf.linspace(-math.pi, math.pi, output_width)
  pitch = tf.linspace(0.0, math.pi, output_height)
  heading, pitch = tf.meshgrid(heading, pitch)
  xs = tf.sin(pitch) * tf.sin(heading)
  ys = -tf.cos(pitch)
  zs = tf.sin(pitch) * tf.cos(heading)
  pixel_rays = tf.reshape(tf.stack([xs, ys, zs], axis=0), ((3, -1)))
  return pixel_rays


def project_to_feat(
    transformed_coords: tf.Tensor,
    feats: tf.Tensor,
    height: int,
    width: int,
    depth_scale: float,
    input_void_class: float,
    output_void_class: float = 0,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Project a set of features to a transformed coordinate space.

  Args:
    transformed_coords: Tensor of shape (N, 4, M) of (x, y, z, 1) values, where
      M denotes the number of data points in the point cloud.
    feats: Tensor of shape (N, M) or (N, M, C) of feature values corresponding
      to each location.
    height: Image height in pixels.
    width: Image width in pixels.
    depth_scale: Maximum depth in meters. Values above this are clipped.
    input_void_class: Feature value (class label) that represents an invalid
      point in the input feats.
    output_void_class: Feature value to use in output projected_feat for an
      invalid pixel. By default 0 is used since this corresponds to the void
      class for Matterport segmentations and black pixels in an RGB image.

  Returns:
    projected_depth: Tensor of shape (N, H, W) of depth values in [0, 1].
    projected_feat: Tensor of shape (N, H, W) or (N, H, W, C) of projected
      feature values. Output shape is dependent on the input shape.
  """
  if len(feats.shape) != 2 and len(feats.shape) != 3:
    raise ValueError('feats should have shape (N, M) or (N, M, C), got'
                     f' {feats.shape} instead.')
  is_scalar_feat = len(feats.shape) == 2
  if is_scalar_feat:
    feats = feats[..., None]  # Unsqueeze last dimension to act as a channel.
  channels = feats.shape[-1]
  batch_size = transformed_coords.shape[0]
  # Normalize x, y values by depth.
  depth = transformed_coords[:, 2, :]
  view_coords = tf.math.divide_no_nan(
      transformed_coords[:, 0:2, :], depth[:, None, ...])
  dtype = transformed_coords.dtype

  # Find all valid coordinates.
  denorm_coords = tf.cast(
      tf.stack([(view_coords[:, 0, :] + 1) / 2 * tf.cast(width, dtype),
                (view_coords[:, 1, :] + 1) / 2 * tf.cast(height, dtype)],
               axis=1), tf.int32)
  valid_coords = tf.math.logical_and(
      tf.math.logical_and(denorm_coords[:, 0, :] >= 0,
                          denorm_coords[:, 0, :] < width),
      tf.math.logical_and(denorm_coords[:, 1, :] >= 0,
                          denorm_coords[:, 1, :] < height))
  # Exclude points that are behind the camera or have no depth return.
  valid_coords = tf.math.logical_and(valid_coords, depth > 0)
  # Exclude points that are void class.
  valid_feats = tf.reduce_all(feats != input_void_class, axis=-1)
  valid_coords = tf.math.logical_and(valid_coords, valid_feats)
  # Convert to a 1D tensor for scattering.
  batch_offset = tf.range(0, batch_size)[:, None] * width * height
  flat_coords = (batch_offset + denorm_coords[:, 1, :] * width +
                 denorm_coords[:, 0, :]) * tf.cast(valid_coords, tf.int32)
  flat_coords = tf.reshape(flat_coords, (-1,))
  flat_depth = tf.reshape(depth, (-1,))

  # Calculate reprojected depth image
  scattered_depth = tf.tensor_scatter_nd_min(
      tf.cast(tf.fill((batch_size * height * width, 1), depth_scale), dtype),
      flat_coords[:, None], flat_depth[..., None])
  projected_depth = tf.reshape(scattered_depth, (batch_size, height, width))
  projected_depth = tf.clip_by_value(
      projected_depth, 0, depth_scale) / depth_scale

  # A lot of points in the cloud collide when mapped to pixel space.
  # Gather from the depth map to identify which points had the minimum depth
  # in pixel space, discard the others to avoid collisions when reprojecting
  # segmentation classes.
  min_depth = tf.gather(scattered_depth, flat_coords)[..., 0]
  flat_coords = flat_coords * tf.cast(flat_depth < min_depth + 0.1, tf.int32)

  # Calculate reprojected feature image
  flat_feats = tf.reshape(feats, (-1, channels))
  scattered_feat = tf.tensor_scatter_nd_max(
      tf.cast(
          tf.fill((batch_size * height * width, channels), output_void_class),
          dtype), flat_coords[:, None], flat_feats)
  projected_feat = tf.reshape(scattered_feat,
                              (batch_size, height, width, channels))

  # Remove channels dimension if initial feature was scalar.
  if is_scalar_feat:
    projected_feat = projected_feat[..., 0]
  return projected_depth, projected_feat


def project_feats_to_equirectangular(
    feats: tf.Tensor, xyz1: tf.Tensor, height: int, width: int,
    void_class: float,
    depth_scale: float,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Project point cloud feats / coords into an equirectangular image.

  Args:
    feats: (N, M) or (N, M, C) tensor of semantic segmentation features.
    xyz1: (N, 4, M): xyz1 coordinates.
    height: Height in pixels of the projected image.
    width: Width in pixels of the projected image.
    void_class: Feature value (class label) that represents an empty pixel.
    depth_scale: Maximum depth in meters. Values above this are clipped.

  Returns:
    reprojected_depth: (N, H, W) tensor of equirectangular image of depth
      values in [0, 1].
    reprojected_feats: (N, H, W) tensor of equirectangular image of features.
  """
  # Map frame coords to the new location.
  # relative_pos indicates the relative displacement of frame i wrt frame 0.
  x, y, z = xyz1[:, 0, :], xyz1[:, 1, :], xyz1[:, 2, :]
  rad = (x**2 + y**2 + z**2)**0.5
  # Heading as defined from the x-axis, which is between the center of the pano
  # image and the right hand edge and rotating left.
  heading = tf.math.atan2(y, x)
  # Heading redefined from the left hand edge of the image and rotating right.
  heading = 1.5 * math.pi - heading
  dtype = xyz1.dtype
  # Map to [0, 2pi] domain.
  heading = heading + (2 * math.pi) * tf.cast(heading <= 0, dtype)
  heading = heading - (2 * math.pi) * tf.cast(heading > (2 * math.pi), dtype)
  elevation = tf.math.acos(tf.math.divide_no_nan(z, rad))

  # Map to 360 panorama image coordinates.
  proj_x = rad * ((heading / (2 * math.pi)) * 2 - 1)
  proj_y = rad * ((elevation / math.pi) * 2 - 1)
  proj_z = rad
  proj_xyz1 = tf.stack([proj_x, proj_y, proj_z, tf.ones_like(proj_x)], axis=1)

  reprojected_depth, reprojected_feats = project_to_feat(
      tf.cast(proj_xyz1, dtype), tf.cast(feats, dtype), height, width,
      depth_scale=depth_scale, input_void_class=void_class)
  return reprojected_depth, reprojected_feats


def equirectangular_to_pointcloud(
    feats: tf.Tensor,
    depth: tf.Tensor,
    void_class: float,
    depth_scale: float,
    size_mult: float = 1.0,
    interpolation_method: str = 'nearest') -> Tuple[tf.Tensor, tf.Tensor]:
  """Filter and return valid coords and features for equirectangular image.

  Coordinates/features that are not visible due to invalid depth are still
  returned but given a feature value of void_class and an xyz1 coordinate of
  (0, 0, 0, 1).

  Args:
    feats: (N, H, W) or (N, H, W, C) tensor of feature values.
    depth: (N, H, W) tensor of depth values, with values in [0, 1].
    void_class: feature value to use for invalid points in the output.
    depth_scale: Maximum depth in metres.
    size_mult: Amount of upscale the features / depths by. This creates denser
      point clouds.
    interpolation_method: Interpolation method for resizing features when
      size_mult != 1.0.
  Returns:
    xyz1: (N, 4, H * W) tensor of (x, y, z, 1) coordinate values.
    filtered_feats: (N, H * W) or (N, H * W, C) tensor of filtered features.
  """
  if len(feats.shape) != 3 and len(feats.shape) != 4:
    raise ValueError('feats should have shape (N, H, W) or (N, H, W, C),'
                     f' got {feats.shape} instead.')
  if void_class < 0.0 and feats.dtype in [
      tf.uint8, tf.uint16, tf.uint32, tf.uint64
  ]:
    raise ValueError(
        'feats datatype must be signed if the void class is negative')
  is_scalar_feat = len(feats.shape) == 3
  if is_scalar_feat:
    feats = feats[..., None]  # Unsqueeze last dimension to act as a channel.
  batch_size, height, width, channels = feats.shape
  assert width == 2 * height, 'Expected equirectangular input images'
  scaled_height = int(height * size_mult)
  scaled_width = int(width * size_mult)
  pano_depth = tf.image.resize(
      depth[..., None], (scaled_height, scaled_width), method='nearest')[..., 0]
  pano_feats = tf.image.resize(
      feats, (scaled_height, scaled_width), method=interpolation_method)
  dtype = depth.dtype
  # Add points to point cloud memory.
  half_pixel_width = 0.5 * np.pi / scaled_height
  elevation = tf.cast(
      tf.linspace(half_pixel_width, np.pi - half_pixel_width, scaled_height),
      dtype)
  # Define heading from the x-axis, increasing towards the y-axis.
  heading = tf.cast(
      tf.linspace(1.5 * np.pi - half_pixel_width,
                  -0.5 * np.pi + half_pixel_width, scaled_width), dtype)
  # Mask out invalid depths.
  depth_mask = tf.cast(tf.math.logical_and(pano_depth > 0, pano_depth < 1.0),
                       dtype)
  rad = (pano_depth * depth_scale) * depth_mask
  pano_feats = tf.where(depth_mask[..., None] == 0, void_class, pano_feats)

  # Move to correct relative position.
  x = rad * tf.math.sin(elevation)[:, None] * tf.math.cos(heading)[None, :]
  y = rad * tf.math.sin(elevation)[:, None] * tf.math.sin(heading)[None, :]
  z = rad * tf.math.cos(elevation)[:, None]
  xyz1 = tf.stack([
      tf.reshape(x, (batch_size, -1)),
      tf.reshape(y, (batch_size, -1)),
      tf.reshape(z, (batch_size, -1)),
      tf.ones(
          (batch_size, scaled_height * scaled_width),
          dtype=dtype)
  ], axis=1)
  filtered_feats = tf.reshape(pano_feats, (batch_size, -1, channels))

  # Remove channels dimension if initial feature was scalar.
  if is_scalar_feat:
    filtered_feats = filtered_feats[..., 0]
  return xyz1, filtered_feats


def project_perspective_image(image,
                              fov,
                              output_height,
                              camera_intrinsics=None,
                              rotations=None,
                              camera_extrinsics=None,
                              pad_mode='constant',
                              pad_value=0.0,
                              round_to_nearest=False):
  """Converts a perspective to an equirectangular image.

  Modified from //experimental/earthsea/wanderer/geometry_utils.py.

  Args:
    image: Tensor with shape [height, width, channels].
    fov: tensor with the fields of view of the image (vertical, horizontal) in
      radians.
    output_height: Int height of the output image, the width will be double.
    camera_intrinsics: Optional camera intrinsics matrix to use instead of
      computing it using field of view.
    rotations: optional tensor containing pitch and heading in radians for
      rotating camera. A positive pitch rotates the camera upwards. A positive
      heading rotates the camera along the equator clockwise.
    rotation_matrix: Optional 3x3 rotation matrix to use instead as an
      alternative to the rotations parameter.
    pad_mode: Padding mode, one of {`reflect`, `mean`, `constant`}.
    pad_value: value to use if pad_mode is set to constant.
    round_to_nearest: if set to True, coordinates are rounded to integers before
      interpolation. This can be useful, for instance, for labels in semantic
      segmentation.

  Returns:
    output_image: an [output_height, output_height * 2, channels] tensor
      containing the output image.
  """
  assert pad_mode in {'reflect', 'constant',
                      'mean'}, ('Unsupported pad mode: %s' % pad_mode)
  image = tf.expand_dims(image, 0)
  output_width = 2 * output_height

  # Get world coordinates of the points of interest.
  # Longitude range is determined by the longitude of the image corners.
  world_coordinates = equirectangular_pixel_rays(output_height)
  
  
  
  
  # Convert world to image coordinates.
  image_shape = tf.cast(tf.shape(image), tf.float32)
  world_to_image = get_world_to_image_transform(
      (image_shape[1], image_shape[2]), fov,
      camera_intrinsics=camera_intrinsics, rotations=rotations,
      rotation_matrix=camera_extrinsics)
  image_coordinates = world_to_image @ world_coordinates
  # R = camera_extrinsics[:3, :3]
  # T = camera_extrinsics[:3, 3:]
  # image_coordinates = camera_intrinsics @ np.linalg.inv(R) @ (world_coordinates)
  
  
  # tf.reduce_min(image_coordinates), tf.reduce_max(image_coordinates):
  # -5111.6733, 0.14465144
  # -1249.6674, 1249.6674

  
  image_coordinates = tf.transpose(image_coordinates)
  xs_and_ys = image_coordinates[:, :2]
  zs = image_coordinates[:, 2:]
  image_coordinates = tf.where(
      tf.broadcast_to(zs > 0, [output_height * output_width, 2]),
      xs_and_ys / zs, -1 * tf.ones_like(xs_and_ys))
  if round_to_nearest:
    image_coordinates = tf.math.round(image_coordinates)

  # Interpolate.
  if pad_mode != 'reflect':
    constant_values = tf.reduce_mean(image) if pad_mode == 'mean' else pad_value
    image = tf.pad(
        image, ((0, 0), (1, 1), (1, 1), (0, 0)),
        mode='constant',
        constant_values=constant_values)
    image_coordinates = image_coordinates + 1.  # Account for padding.
  
  # (1, 1026, 1282, 3) (1, 8388608, 2) -> (1, 8388608, 3)
  output_image = tfa_image.interpolate_bilinear(
      image, tf.expand_dims(image_coordinates, 0), indexing='xy')
  output_image = tf.reshape(output_image, [output_height, output_width, -1])
  num_channels = image.shape[-1]
  output_image = tf.ensure_shape(output_image, [None, None, num_channels])
  return output_image


def _xyz_to_lonlat(xyz):
  """Converts the world coordinates into longitude, latitudes."""
  norm = tf.linalg.norm(xyz, axis=-1, keepdims=True)
  xyz_norm = xyz / norm
  x = xyz_norm[..., 0:1]
  y = xyz_norm[..., 1:2]
  z = xyz_norm[..., 2:]

  lon = tf.math.atan2(x, z)
  lat = tf.math.asin(y)
  lst = [lon, lat]

  out = tf.concat(lst, axis=-1)
  return out


def _lonlat_to_uv(lonlat, shape):
  """Converts the longitude/latitudes to image coords with a given shape."""
  u = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
  v = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
  return tf.concat([u, v], axis=-1)


def get_perspective_from_equirectangular_image(image, camera_intrinsics,
                                               rotation_matrix, height, width):
  """Converts the equirectangular image to perspective image.

  Args:
    image: Equirectangular image of shape (H, W, 3)
    camera_intrinsics: camera intrinsic matrix of perspective camera
    rotation_matrix: rotation matrix associated with camera
    height: height of the perspective image
    width: width of the perspective image
  Returns:
    perspective_image: returns the perspective image
  """
  eq_height, eq_width, channels = image.shape

  x = tf.range(width)
  y = tf.range(height)
  x, y = tf.meshgrid(x, y)
  z = tf.ones_like(x)
  xyz = tf.concat([x[..., None], y[..., None], z[..., None]], axis=-1)
  xyz = tf.cast(xyz, tf.float32)
  xyz = (xyz @ tf.transpose(tf.linalg.inv(camera_intrinsics))) @ rotation_matrix

  lonlat = _xyz_to_lonlat(xyz)
  uv = _lonlat_to_uv(lonlat, shape=(eq_height, eq_width))
  uv = tf.cast(uv, tf.float32)
  uv = tf.reshape(uv, (-1, 2))

  image = tf.cast(tf.expand_dims(image, 0), tf.float32)
  perspective_image = tfa_image.interpolate_bilinear(
      image, tf.expand_dims(uv, 0), indexing='xy')
  perspective_image = tf.reshape(perspective_image, [height, width, channels])

  return perspective_image

