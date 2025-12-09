import os, json
import numpy as np
import open3d as o3d
import shutil
from phase3_export import create_html_viewer

def load_ply(folder):
    ply = os.path.join(folder, "sequence_12views_sparse_filtered.ply")
    if not os.path.exists(ply):
        raise FileNotFoundError(f"Missing: {ply}")
    return o3d.io.read_point_cloud(ply)

def load_cameras(folder):
    p = os.path.join(folder, "cameras.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def umeyama_similarity(src, dst):
    """
    Find s, R, t such that:  dst ~ s * R * src + t
    src, dst: Nx3
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape and src.shape[0] >= 3

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    X = src - mu_s
    Y = dst - mu_d

    cov = (Y.T @ X) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    var_s = (X**2).sum() / src.shape[0]
    scale = np.trace(np.diag(D) @ S) / (var_s + 1e-12)
    t = mu_d - scale * (R @ mu_s)
    return float(scale), R, t

def pick_points(pcd, title):
    print("\n" + "="*60)
    print(title)
    print("Shift + Left Click to pick points, then press 'Q' to finish.")
    print("Pick at least 4 points, IN THE SAME ORDER in both clouds.")
    print("="*60)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=title, width=1280, height=720)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    return vis.get_picked_points()

def transform_camera_block(cam_block, scale, R, t, id_offset=0):
    cams = cam_block["cameras"]
    out = []
    for c in cams:
        C = np.array([c["position"]["x"], c["position"]["y"], c["position"]["z"]], dtype=np.float64)
        C2 = scale * (R @ C) + t

        c2 = dict(c)
        c2["id"] = int(c["id"]) + id_offset
        c2["image_id"] = int(c.get("image_id", c["id"])) # keep image_id stable
        c2["position"] = {"x": float(C2[0]), "y": float(C2[1]), "z": float(C2[2])}

        # rotate orientation too if quaternion exists
        if "quaternion" in c2:
            # convert R to quaternion and left-multiply: q_new = q_R * q_old
            qR = rotmat_to_quat_xyzw(R)
            q = np.array([c2["quaternion"]["x"], c2["quaternion"]["y"], c2["quaternion"]["z"], c2["quaternion"]["w"]], dtype=np.float64)
            qn = quat_mul(qR, q)
            c2["quaternion"] = {"x": float(qn[0]), "y": float(qn[1]), "z": float(qn[2]), "w": float(qn[3])}

        out.append(c2)
    return out

def rotmat_to_quat_xyzw(R):
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0,0], R[1,1], R[2,2]]))
        if i == 0:
            s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            w = (R[2,1] - R[1,2]) / s
            x = 0.25*s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25*s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25*s
    q = np.array([x,y,z,w], dtype=np.float64)
    q /= (np.linalg.norm(q) + 1e-12)
    return q

def quat_mul(q1, q2):
    x1,y1,z1,w1 = q1
    x2,y2,z2,w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=np.float64)

def main(folderA="Ply_pose1_DLC", folderB="Ply_pose2_DLC", out="Ply_merged_DLC"):
    pcdA = load_ply(folderA)
    pcdB = load_ply(folderB)

    # downsample for easier picking
    pcdA_ds = pcdA.voxel_down_sample(voxel_size=max(0.001, np.linalg.norm(np.asarray(pcdA.get_max_bound())-np.asarray(pcdA.get_min_bound()))/300.0))
    pcdB_ds = pcdB.voxel_down_sample(voxel_size=max(0.001, np.linalg.norm(np.asarray(pcdB.get_max_bound())-np.asarray(pcdB.get_min_bound()))/300.0))

    idxA = pick_points(pcdA_ds, f"Pick points in A: {folderA}")
    idxB = pick_points(pcdB_ds, f"Pick points in B: {folderB}")

    if len(idxA) < 3 or len(idxB) < 3 or len(idxA) != len(idxB):
        raise RuntimeError("Need same number of picked points (>=3) in both clouds.")

    PA = np.asarray(pcdA_ds.points)[idxA]
    PB = np.asarray(pcdB_ds.points)[idxB]

    scale, R, t = umeyama_similarity(PB, PA)  # map B -> A
    print(f"\nEstimated similarity: scale={scale:.4f}\nR=\n{R}\nt={t}")

    # transform B point cloud into A frame
    ptsB = np.asarray(pcdB.points)
    ptsB2 = (scale * (R @ ptsB.T)).T + t.reshape(1,3)

    ptsA = np.asarray(pcdA.points)
    merged = np.vstack([ptsA, ptsB2])

    os.makedirs(out, exist_ok=True)

    # save merged PLY
    pcdM = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(merged))
    o3d.io.write_point_cloud(os.path.join(out, "merged_filtered.ply"), pcdM)
    print(f"Saved: {os.path.join(out, 'merged_filtered.ply')}")

    # save merged points.json
    save_json({"positions": merged.flatten().tolist(), "count": int(len(merged))}, os.path.join(out, "points.json"))

    # merge cameras.json
    camsA = load_cameras(folderA)
    camsB = load_cameras(folderB)

    max_id_A = max(int(c["id"]) for c in camsA["cameras"])
    offset = max_id_A + 1

    camsB_xf = transform_camera_block(camsB, scale, R, t, id_offset=offset)
    cams_all = camsA["cameras"] + camsB_xf
    save_json({"cameras": cams_all, "num_cameras": len(cams_all)}, os.path.join(out, "cameras.json"))

    # merge view graphs as disconnected components (still usable)
    vg = {}
    for k, nbrs in json.load(open(os.path.join(folderA, "view_graph.json"), "r", encoding="utf-8")).items():
        vg[str(int(k))] = nbrs
    vgB = json.load(open(os.path.join(folderB, "view_graph.json"), "r", encoding="utf-8"))
    for k, nbrs in vgB.items():
        kk = str(int(k) + offset)
        vg[kk] = [{"target": int(e["target"]) + offset, "shared_points": int(e.get("shared_points", 0))} for e in nbrs]
    save_json(vg, os.path.join(out, "view_graph.json"))

    # make merged folder serveable: add index.html + images
    create_html_viewer(os.path.join(out, "index.html"))

    # copy images (use pose1's images folder as the source)
    src_images = os.path.join(folderA, "images")
    dst_images = os.path.join(out, "images")
    if os.path.exists(src_images):
        shutil.copytree(src_images, dst_images, dirs_exist_ok=True)
        print(f"Copied images -> {dst_images}")
    else:
        print(f"WARNING: No images folder found at {src_images}. (Run phase2 export with images enabled.)")


    print(f"\nMerged export written to: {out}")
    print(f"Host it: cd {out} && python -m http.server 8000")

if __name__ == "__main__":
    main()
