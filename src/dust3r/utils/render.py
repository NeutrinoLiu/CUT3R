import torch
from gsplat import rasterization
from dust3r.utils.geometry import inv, geotrf
from dust3r.utils.camera import pose_encoding_to_camera

# ------------------------------ point renderer ------------------------------ #
def render_point_cloud(points, colors, intrinsics, width, height):
    """
    Renders a colored point cloud to a 2D image using PyTorch.
    
    Args:
        points: torch tensor of shape (3, N) containing 3D points (x, y, z)
        colors: torch tensor of shape (3, N) containing RGB colors for each point
        cam_intrinsics: torch tensor of shape (3, 3) containing camera intrinsic matrix
        width: int, width of the output image in pixels
        height: int, height of the output image in pixels
        device: torch device to use (defaults to points device)
        
    Returns:
        image: torch tensor of shape (height, width, 3) containing the rendered image
    """
    # Use the same device as the input points if not specified
    device = "cuda"
    
    # Initialize empty image
    image = torch.zeros((height, width, 3), dtype=torch.float, device=device)
    
    # Number of points
    n_points = points.shape[1]
    
    # Project 3D points to 2D using the intrinsic matrix
    # Assuming extrinsic matrix is identity (points are already in camera coordinates)
    points_homogeneous = torch.cat([points, torch.ones(1, n_points, device=device)], dim=0)
    pixels_homogeneous = torch.matmul(intrinsics, points_homogeneous[:3])
    
    # Convert to inhomogeneous coordinates
    pixels = pixels_homogeneous[:2] / pixels_homogeneous[2:3]
    
    # Convert to integer pixel coordinates
    pixels = torch.round(pixels).long()
    
    # Filter points that are in front of the camera (z > 0) and within image bounds
    valid_mask = (
        (points[2] > 0) &
        (pixels[0] >= 0) & (pixels[0] < width) &
        (pixels[1] >= 0) & (pixels[1] < height)
    )
    
    valid_pixels = pixels[:, valid_mask]
    valid_colors = colors[:, valid_mask]
    valid_depths = points[2, valid_mask]
    
    # Create a depth buffer to handle occlusion
    depth_buffer = torch.full((height, width), float('inf'), device=device)
    
    # Use a more efficient GPU implementation with scatter
    depth_buffer_flat = depth_buffer.view(-1)
    image_flat = image.view(-1, 3)
    
    # Convert 2D indices to flat indices
    flat_indices = valid_pixels[1] * width + valid_pixels[0]
    
    # Process points in order of increasing depth for correct occlusion
    sorted_indices = torch.argsort(valid_depths)
    flat_indices = flat_indices[sorted_indices]
    valid_colors = valid_colors[:, sorted_indices]
    valid_depths = valid_depths[sorted_indices]
    
    # Use scatter_ to update the depth buffer and image
    depth_buffer_flat.scatter_(0, flat_indices, valid_depths)
    
    # Convert colors to 0-255 range and correct layout for image
    image_flat.scatter_(0, flat_indices.unsqueeze(1).repeat(1, 3), valid_colors.t())
    
    # Reshape back
    image = image_flat.view(height, width, 3)

    return image


def render(
    intrinsics: torch.Tensor,
    pts3d: torch.Tensor,
    rgbs: torch.Tensor | None = None,
    scale: float = 0.002,
    opacity: float = 0.95,
):

    device = pts3d.device
    batch_size = len(intrinsics)
    img_size = pts3d.shape[1:3]
    pts3d = pts3d.reshape(batch_size, -1, 3)
    num_pts = pts3d.shape[1]
    quats = torch.randn((num_pts, 4), device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = scale * torch.ones((num_pts, 3), device=device)
    opacities = opacity * torch.ones((num_pts), device=device)
    if rgbs is not None:
        assert rgbs.shape[1] == 3
        rgbs = rgbs.reshape(batch_size, 3, -1).transpose(1, 2) # (B, N, 3)
    else:
        rgbs = torch.ones_like(pts3d[:, :, :3])

    rendered_rgbs = []
    rendered_depths = []
    accs = []
    point_rgbs = []
    for i in range(batch_size):
        rgbd, acc, _ = rasterization(
            pts3d[i],
            quats,
            scales,
            opacities,
            rgbs[i],
            torch.eye(4, device=device)[None],
            intrinsics[[i]],
            width=img_size[1],
            height=img_size[0],
            packed=False,
            render_mode="RGB+D",
        )
        point_rgb = render_point_cloud(pts3d[i].T, rgbs[i].T, intrinsics[i], img_size[1], img_size[0])

        rendered_depths.append(rgbd[..., 3])
        rendered_rgbs.append(rgbd[..., :3])
        point_rgbs.append(point_rgb[None])

    rendered_depths = torch.cat(rendered_depths, dim=0)
    rendered_rgbs = torch.cat(rendered_rgbs, dim=0)
    point_rgbs = torch.cat(point_rgbs, dim=0)

    # print(f"shape of gs rgbs: {rendered_rgbs.shape}")
    # print(f"shape of point rgbs: {point_rgbs.shape}")
    # print(f"data range of gs rgbs: {rendered_rgbs.min()}, {rendered_rgbs.max()}")
    # print(f"data range of point rgbs: {point_rgbs.min()}, {point_rgbs.max()}")
    # raise ValueError

    return point_rgbs, rendered_rgbs, rendered_depths, accs


def get_render_results(gts, preds, self_view=False, return_pts_rgb=True):
    device = preds[0]["pts3d_in_self_view"].device
    with torch.no_grad():
        depths = []
        gt_depths = []
        imgs = []
        gt_imgs = []
        for i, (gt, pred) in enumerate(zip(gts, preds)):
            if self_view:
                camera = inv(gt["camera_pose"]).to(device)
                intrinsics = gt["camera_intrinsics"].to(device)
                pred = pred["pts3d_in_self_view"]
            else:
                camera = inv(gts[0]["camera_pose"]).to(device)
                intrinsics = gts[0]["camera_intrinsics"].to(device)
                pred = pred["pts3d_in_other_view"]
            gt_img = gt["img"].to(device)
            gt_pts3d = gt["pts3d"].to(device)

            ptimg, gsimg, depth, _ = render(intrinsics, pred, gt_img)
            gt_ptimg, gt_gsimg, gt_depth, _ = render(intrinsics, geotrf(camera, gt_pts3d), gt_img)
            
            # print(f"shape of gsimg: {gsimg[0].shape}")
            # print(f"data range of gsimg: {gsimg[0].min()}, {gsimg[0].max()}")
            # raise ValueError

            depths.append(depth)
            gt_depths.append(gt_depth)
            if return_pts_rgb:
                imgs.append(ptimg)
                gt_imgs.append(gt_ptimg)
            else:
                # return 3dgs rgb
                imgs.append(gsimg)
                gt_imgs.append(gt_gsimg)

    return depths, gt_depths, imgs, gt_imgs

def get_render_results_reproj(gts, preds, return_pts_rgb=True):
    device = preds[0]["pts3d_in_self_view"].device
    with torch.no_grad():
        depths = []
        gt_depths = []
        imgs = []
        gt_imgs = []
        for i, (gt, pred) in enumerate(zip(gts, preds)):
            intrinsics = gt["camera_intrinsics"].to(device)
            gt_rgb = gt["img"].to(device)

            pred_camera = inv(pose_encoding_to_camera(pred["camera_pose"])).to(device)
            pred_pts3d = pred["pts3d_in_other_view"]
            gt_camera = inv(gt["camera_pose"]).to(device)
            gt_pts3d = gt["pts3d"].to(device)

            ptimg, gsimg, depth, _ = render(intrinsics, geotrf(pred_camera, pred_pts3d), gt_rgb)
            gt_ptimg, gt_gsimg, gt_depth, _ = render(intrinsics, geotrf(gt_camera, gt_pts3d), gt_rgb)

            depths.append(depth)
            gt_depths.append(gt_depth)
            if return_pts_rgb:
                imgs.append(ptimg)
                gt_imgs.append(gt_ptimg)
            else:
                # return 3dgs rgb
                imgs.append(gsimg)
                gt_imgs.append(gt_gsimg)
    return depths, gt_depths, imgs, gt_imgs

