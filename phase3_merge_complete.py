# Complete Phase 3 Merge Script - Interactive Point Cloud Alignment

import os
import json
import numpy as np
import open3d as o3d
import shutil
from pathlib import Path


def rotmat_to_quat_xyzw(R):
    """Convert 3x3 rotation matrix to quaternion (x,y,z,w)"""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    
    q = np.array([x, y, z, w], dtype=np.float64)
    q /= (np.linalg.norm(q) + 1e-12)
    return q


def quat_mul(q1, q2):
    """Multiply two quaternions (x,y,z,w format)"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=np.float64)


def umeyama_similarity(src, dst):
    """
    Compute similarity transform (scale, rotation, translation)
    Maps src -> dst using Umeyama's method
    Returns: scale (float), R (3x3), t (3,)
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    
    if src.shape[0] < 3:
        raise ValueError("Need at least 3 corresponding points")
    
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    
    X = src - mu_src
    Y = dst - mu_dst
    
    # covariance matrix
    cov = (Y.T @ X) / src.shape[0]
    
    # SVD
    U, D, Vt = np.linalg.svd(cov)
    
    # handle reflections
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    
    R = U @ S @ Vt
    
    # scale
    var_src = (X**2).sum() / src.shape[0]
    scale = np.trace(np.diag(D) @ S) / (var_src + 1e-12)
    
    # translation
    t = mu_dst - scale * (R @ mu_src)
    
    return float(scale), R, t


def pick_corresponding_points(pcdA, pcdB, titleA="Cloud A", titleB="Cloud B"):
    """
    Interactive point picking in both clouds
    Returns: (points_A, points_B) as Nx3 numpy arrays
    """
    print("\n" + "="*70)
    print(f"STEP 1: Pick corresponding points in {titleA}")
    print("="*70)
    print("Instructions:")
    print("  1. Shift + Left Click to pick a point")
    print("  2. Pick at least 4-6 clearly identifiable landmarks")
    print("  3. Press Q when done")
    print("  4. Remember the ORDER - you'll pick the same points in Cloud B!")
    print("="*70)
    
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=titleA, width=1400, height=900)
    vis.add_geometry(pcdA)
    vis.run()
    vis.destroy_window()
    
    idxA = vis.get_picked_points()
    
    if len(idxA) < 3:
        raise RuntimeError(f"Need at least 3 points, got {len(idxA)}")
    
    print(f"\n✓ Picked {len(idxA)} points in {titleA}")
    
    print("\n" + "="*70)
    print(f"STEP 2: Pick THE SAME {len(idxA)} points in {titleB} (IN THE SAME ORDER!)")
    print("="*70)
    print("Pick the corresponding landmarks in the same sequence.")
    print("="*70)
    
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=titleB, width=1400, height=900)
    vis.add_geometry(pcdB)
    vis.run()
    vis.destroy_window()
    
    idxB = vis.get_picked_points()
    
    if len(idxB) != len(idxA):
        raise RuntimeError(f"Point count mismatch: {len(idxA)} in A, {len(idxB)} in B")
    
    print(f"\n✓ Picked {len(idxB)} points in {titleB}")
    
    ptsA = np.asarray(pcdA.points)[idxA]
    ptsB = np.asarray(pcdB.points)[idxB]
    
    return ptsA, ptsB


def transform_camera_pose(R_wc, t_wc, scale, R_sim, t_sim):
    """
    Transform camera extrinsics (world->cam) with similarity transform
    
    Args:
        R_wc: 3x3 rotation (world to camera)
        t_wc: 3x1 translation vector
        scale: scalar scale factor
        R_sim: 3x3 rotation of similarity transform
        t_sim: 3x1 translation of similarity transform
    
    Returns:
        R_new, t_new (transformed extrinsics)
    """
    R_wc = np.asarray(R_wc, dtype=np.float64)
    t_wc = np.asarray(t_wc, dtype=np.float64).reshape(3, 1)
    
    # camera center in world coords
    C_old = -R_wc.T @ t_wc
    C_old = C_old.reshape(3)
    
    # transform camera center
    C_new = scale * (R_sim @ C_old) + t_sim
    
    # transform camera orientation
    R_new = R_sim @ R_wc
    
    # recompute translation vector
    t_new = -R_new @ C_new.reshape(3, 1)
    
    return R_new, t_new


def merge_two_reconstructions(folderA, folderB, output_folder, min_shared=30):
    """
    Main merge function - combines two SfM reconstructions
    
    Args:
        folderA: path to first reconstruction (reference frame)
        folderB: path to second reconstruction (will be transformed)
        output_folder: where to save merged result
        min_shared: minimum shared points for view graph edges
    """
    print("\n" + "="*70)
    print("PHASE 3 RECONSTRUCTION MERGE")
    print("="*70)
    
    # 1. load point clouds
    print("\n[1/7] Loading point clouds...")
    plyA = os.path.join(folderA, "sequence_12views_sparse_filtered.ply")
    plyB = os.path.join(folderB, "sequence_12views_sparse_filtered.ply")
    
    if not os.path.exists(plyA):
        raise FileNotFoundError(f"Missing: {plyA}")
    if not os.path.exists(plyB):
        raise FileNotFoundError(f"Missing: {plyB}")
    
    pcdA = o3d.io.read_point_cloud(plyA)
    pcdB = o3d.io.read_point_cloud(plyB)
    
    print(f"  Cloud A: {len(pcdA.points)} points")
    print(f"  Cloud B: {len(pcdB.points)} points")
    
    # colour for visualisation
    pcdA.paint_uniform_color([1.0, 0.7, 0.0]) # Orange
    pcdB.paint_uniform_color([0.0, 0.6, 0.9]) # Blue
    
    # downsample for easier picking
    print("\n[2/7] Downsampling for point picking...")
    extentA = np.linalg.norm(np.asarray(pcdA.get_max_bound()) - np.asarray(pcdA.get_min_bound()))
    extentB = np.linalg.norm(np.asarray(pcdB.get_max_bound()) - np.asarray(pcdB.get_min_bound()))
    
    voxelA = max(0.002, extentA / 400.0)
    voxelB = max(0.002, extentB / 400.0)
    
    pcdA_ds = pcdA.voxel_down_sample(voxel_size=voxelA)
    pcdB_ds = pcdB.voxel_down_sample(voxel_size=voxelB)
    
    # 2. interactive point picking
    print("\n[3/7] Interactive point correspondence selection...")
    try:
        ptsA, ptsB = pick_corresponding_points(
            pcdA_ds, pcdB_ds,
            titleA=f"REFERENCE: {folderA}",
            titleB=f"TO ALIGN: {folderB}"
        )
    except Exception as e:
        print(f"\nPoint picking failed: {e}")
        return False
    
    # 3. compute similarity transform
    print("\n[4/7] Computing similarity transformation...")
    scale, R, t = umeyama_similarity(ptsB, ptsA)  # map B -> A
    
    print("\nEstimated transformation:")
    print(f"Scale: {scale:.4f}")
    print(f"Rotation:\n{R}")
    print(f"Translation: {t}")
    
    # build 4x4 matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    # 4. transform and merge point clouds
    print("\n[5/7] Transforming and merging point clouds")
    ptsB_orig = np.asarray(pcdB.points)
    ptsB_transformed = (scale * (R @ ptsB_orig.T)).T + t.reshape(1, 3)
    
    ptsA_orig = np.asarray(pcdA.points)
    merged_points = np.vstack([ptsA_orig, ptsB_transformed])
    
    os.makedirs(output_folder, exist_ok=True)
    
    # save merged PLY
    pcdM = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(merged_points))
    merged_ply_path = os.path.join(output_folder, "sequence_12views_sparse_filtered.ply")
    o3d.io.write_point_cloud(merged_ply_path, pcdM)
    print(f"  ✓ Saved: {merged_ply_path}")
    
    # save points.json for viewer
    points_json = {
        "positions": merged_points.astype(float).flatten().tolist(),
        "count": int(len(merged_points))
    }
    with open(os.path.join(output_folder, "points.json"), "w") as f:
        json.dump(points_json, f)
    
    # 5. transform and merge cameras
    print("\n[6/7] Transforming and merging cameras...")
    
    # load camera data
    with open(os.path.join(folderA, "cameras.json"), "r") as f:
        camsA = json.load(f)
    with open(os.path.join(folderB, "cameras.json"), "r") as f:
        camsB = json.load(f)
    
    # Find max id in A to offset B
    max_id_A = max(int(c["id"]) for c in camsA["cameras"])
    id_offset = max_id_A + 1
    
    # transform cameras from B
    quat_R = rotmat_to_quat_xyzw(R)
    camsB_transformed = []
    
    for cam in camsB["cameras"]:
        # transform position (camera center)
        C_old = np.array([cam["position"]["x"], cam["position"]["y"], cam["position"]["z"]])
        C_new = scale * (R @ C_old) + t
        
        # transform quaternion (orientation)
        q_old = np.array([
            cam["quaternion"]["x"],
            cam["quaternion"]["y"],
            cam["quaternion"]["z"],
            cam["quaternion"]["w"]
        ])
        q_new = quat_mul(quat_R, q_old)
        
        # Also transform raw extrinsics if present
        R_wc_new, t_wc_new = None, None
        if "matrix_R_wc" in cam and "vector_t_wc" in cam:
            R_wc = np.array(cam["matrix_R_wc"])
            t_wc = np.array(cam["vector_t_wc"])
            R_wc_new, t_wc_new = transform_camera_pose(R_wc, t_wc, scale, R, t)
        
        cam_new = {
            "id": int(cam["id"]) + id_offset,
            "image_id": int(cam.get("image_id", cam["id"])),
            "image": cam["image"],
            "position": {"x": float(C_new[0]), "y": float(C_new[1]), "z": float(C_new[2])},
            "quaternion": {"x": float(q_new[0]), "y": float(q_new[1]), "z": float(q_new[2]), "w": float(q_new[3])}
        }
        
        if R_wc_new is not None:
            cam_new["matrix_R_wc"] = R_wc_new.tolist()
            cam_new["vector_t_wc"] = t_wc_new.reshape(3).tolist()
        
        camsB_transformed.append(cam_new)
    
    # merge camera lists
    all_cameras = {
        "cameras": camsA["cameras"] + camsB_transformed,
        "num_cameras": len(camsA["cameras"]) + len(camsB_transformed)
    }
    
    with open(os.path.join(output_folder, "cameras.json"), "w") as f:
        json.dump(all_cameras, f, indent=2)
    
    print(f"  ✓ Merged {len(camsA['cameras'])} + {len(camsB_transformed)} cameras")
    
    # 6. merge view graphs (as disconnected components initially)
    print("\n[7/7] Merging view graphs...")
    
    with open(os.path.join(folderA, "view_graph.json"), "r") as f:
        vgA = json.load(f)
    with open(os.path.join(folderB, "view_graph.json"), "r") as f:
        vgB = json.load(f)
    
    merged_vg = {}
    
    # add A's graph
    for k, neighbors in vgA.items():
        merged_vg[str(int(k))] = neighbors
    
    # add B's graph with offset IDs
    for k, neighbors in vgB.items():
        new_k = str(int(k) + id_offset)
        new_neighbors = [
            {
                "target": int(n["target"]) + id_offset,
                "shared_points": int(n.get("shared_points", 0))
            }
            for n in neighbors
        ]
        merged_vg[new_k] = new_neighbors
    
    with open(os.path.join(output_folder, "view_graph.json"), "w") as f:
        json.dump(merged_vg, f, indent=2)
    
    print(f"Merged view graphs complete")
    
    # 7. copy images
    print("\nCopying images to viewer folder...")
    imgs_out = os.path.join(output_folder, "images")
    os.makedirs(imgs_out, exist_ok=True)
    
    for src_folder in [folderA, folderB]:
        src_imgs = os.path.join(src_folder, "images")
        if os.path.exists(src_imgs):
            for img_file in os.listdir(src_imgs):
                src = os.path.join(src_imgs, img_file)
                dst = os.path.join(imgs_out, img_file)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
    
    print(f"Images copied to {imgs_out}")
    
    # 8. create HTML viewer
    from phase3_export import create_html_viewer
    create_html_viewer(os.path.join(output_folder, "index.html"))
    
    # success summary
    print("\n" + "="*70)
    print("MERGE COMPLETE!")
    print("="*70)
    print(f"Output folder: {output_folder}")
    print(f"Total cameras: {len(all_cameras['cameras'])}")
    print(f"Total points: {len(merged_points)}")
    print("\nTo view:")
    print(f"  cd {output_folder}")
    print(f"  python -m http.server 8000")
    print(f"  Open: http://localhost:8000/index.html")
    print("="*70)
    
    return True


if __name__ == "__main__":
    # example usage
    folderA = "Ply_pose1_DLC" # reference reconstruction
    folderB = "Ply_pose2_DLC" # reconstruction to align
    output = "Ply_merged_DLC" # Output folder
    
    success = merge_two_reconstructions(folderA, folderB, output)
    
    if not success:
        print("\nMerge failed. Please try again.")