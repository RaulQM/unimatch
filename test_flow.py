from PIL import Image
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

@torch.no_grad()
def plot_optical_flow(flow, step=10, scale=1.0):
    """
    Plots the optical flow field as arrows.

    Parameters:
        flow (numpy.ndarray): Optical flow field of shape (H, W, 2), where:
                             - flow[..., 0] is the horizontal displacement (u),
                             - flow[..., 1] is the vertical displacement (v).
        step (int): Sampling step size to reduce the number of arrows plotted for clarity.
        scale (float): Scaling factor for the arrow lengths.

    Returns:
        None
    """
    H, W, _ = flow.shape  # Get dimensions of the flow field
    y, x = np.mgrid[0:H:step, 0:W:step]  # Grid of coordinates for sampling
    u = flow[::step, ::step, 0]  # Horizontal component (u) sampled
    v = flow[::step, ::step, 1]  # Vertical component (v) sampled

    plt.figure(figsize=(10, 10))
    plt.imshow(np.zeros((H, W)), cmap="gray")  # Background (black image)
    plt.quiver(x, y, u, -v, angles='xy', scale_units='xy', scale=1.0/scale, color='red')  # Arrows for flow
    plt.title("Optical Flow Visualization")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.axis('on')
    plt.show()

@torch.no_grad()
def flow_to_correspondences(flow, idxs):
    """
    Converts an optical flow field into source and target keypoint correspondences.

    Parameters:
        flow (numpy.ndarray): Optical flow field of shape (H, W, 2), where:
                              - flow[..., 0] is the horizontal displacement (u),
                              - flow[..., 1] is the vertical displacement (v).
        idxs (numpy.ndarray): Corresponding keypoints (N x 2 array of x, y coordinates).

    Returns:
        mkpts0 (numpy.ndarray): N x 2 array of source keypoints (x, y) coordinates.
        mkpts1 (numpy.ndarray): N x 2 array of target keypoints (x + u, y + v) coordinates.
    """
    u_flat = flow[..., 0]
    v_flat = flow[..., 1]

    # Get target coordinates by adding the flow to the source keypoints
    x_target = idxs[:, 1] + u_flat[idxs[:, 0], idxs[:, 1]]  # Target x-coordinates
    y_target = idxs[:, 0] + v_flat[idxs[:, 0], idxs[:, 1]]  # Target y-coordinates

    mkpts0 = np.stack((idxs[:, 1], idxs[:, 0]), axis=1)  # Source keypoints
    mkpts1 = np.stack((x_target, y_target), axis=1)  # Target keypoints

    return mkpts0, mkpts1

@torch.no_grad()
def flow_estimation(model,
                   img0,
                   img1,
                   mask0,
                   padding_factor=8,
                   inference_size=None,
                   attn_type='swin',
                   attn_splits_list=None,
                   corr_radius_list=None,
                   prop_radius_list=None,
                   num_reg_refine=1,
                   pred_bidir_flow=False,
                   pred_bwd_flow=False,
                   fwd_bwd_consistency_check=False,
                   ):
    """ Inference on a directory or a video """
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if fwd_bwd_consistency_check:
        assert pred_bidir_flow

    fixed_inference_size = inference_size
    transpose_img = False

    if len(img0.shape) == 2:  # gray image
        image0 = np.tile(img0[..., None], (1, 1, 3))
        image1 = np.tile(img1[..., None], (1, 1, 3))
    else:
        image0 = img0[..., :3]
        image1 = img1[..., :3]

    image0 = torch.from_numpy(image0).permute(2, 0, 1).float().unsqueeze(0).to(device)
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).to(device)

    # the model is trained with size: width > height
    if image0.size(-2) > image0.size(-1):
        image0 = torch.transpose(image0, -2, -1)
        image1 = torch.transpose(image1, -2, -1)
        transpose_img = True

    nearest_size = [int(np.ceil(image0.size(-2) / padding_factor)) * padding_factor,
                    int(np.ceil(image0.size(-1) / padding_factor)) * padding_factor]

    # resize to nearest size or specified size
    inference_size = nearest_size if fixed_inference_size is None else fixed_inference_size

    assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
    ori_size = image1.shape[-2:]

    # resize before inference
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        image0 = F.interpolate(image0, size=inference_size, mode='bilinear',
                                align_corners=True)
        image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                align_corners=True)

    if pred_bwd_flow:
        image0, image1 = image1, image0

    results_dict = model(image0, image1,
                            attn_type=attn_type,
                            attn_splits_list=attn_splits_list,
                            corr_radius_list=corr_radius_list,
                            prop_radius_list=prop_radius_list,
                            num_reg_refine=num_reg_refine,
                            task='flow',
                            pred_bidir_flow=pred_bidir_flow,
                            )

    flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]

    # resize back
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                align_corners=True)
        flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
        flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

    if transpose_img:
        flow_pr = torch.transpose(flow_pr, -2, -1)

    flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

    flow[:, :, 0] = flow[:, :, 0] * mask0
    flow[:, :, 1] = flow[:, :, 1] * mask0

    # plot_optical_flow(flow)

    # Image dimensions
    mflow = np.argwhere(mask0 == 1.0)
    miny, minx, maxy, maxx = np.min(mflow[:, 0]), np.min(mflow[:, 1]), np.max(mflow[:, 0]), np.max(mflow[:, 1])
    height = maxy - miny 
    width  = maxx - minx 

    num_samples = 4096

    if num_samples > (height * width):
        num_samples = height * width

    # Create a grid of target positions for uniform sampling
    grid_y = np.linspace(miny, maxy, num=int(np.sqrt(num_samples)), dtype=int)
    grid_x = np.linspace(minx, maxx, num=int(np.sqrt(num_samples)), dtype=int)
    grid = np.array(np.meshgrid(grid_y, grid_x)).T.reshape(-1, 2)
    
    mkpts0, mkpts1 = flow_to_correspondences(flow, grid)

    # # Sampling a subset (let's say 3 random points)
    # sample_size = 1024
    # indices = np.random.choice(len(mkpts0), size=sample_size, replace=False)
    # # Select the sampled correspondences
    # sampled_mkpts0 = mkpts0[indices, :]
    # sampled_mkpts1 = mkpts1[indices, :]
    # # Create a figure to plot
    # fig, ax = plt.subplots(1, 1,)
    # # Concatenate images side by side for better visualization
    # img_concat = np.hstack((img0, img1))
    # # Plot the concatenated image
    # ax.imshow(img_concat)
    # # Adjust the keypoints in img2 to account for the concatenated layout
    # keypoints_img2_adjusted = [(x + img0.shape[1], y) for (x, y) in sampled_mkpts1]
    # # Plot keypoints and lines
    # for (pt1, pt2) in zip(sampled_mkpts0, keypoints_img2_adjusted):
    #     # Plot circles at the keypoints
    #     ax.plot(pt1[0], pt1[1], 'wo', markersize=3)  # Green circles in image1
    #     ax.plot(pt2[0], pt2[1], 'wo', markersize=3)  # Green circles in image2
    #     # Plot lines between keypoints
    #     ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'w-', linewidth=1)  # Blue lines between correspondences
    # # Hide the axes
    # ax.axis('off')
    # # Show the plot
    # plt.show()

    return mkpts0, mkpts1
