import open3d as o3d
import numpy as np
import json
from pathlib import Path

ROOT = Path(".").resolve()

# merge point clouds
def merge_and_save():
    # Load partial clouds
    pcd_A = o3d.io.read_point_cloud("wall_A_filtered.ply")
    pcd_B = o3d.io.read_point_cloud("wall_B_filtered.ply")

    # got to define these manually after testing
    theta = np.radians(90)  # example: rotate 90° about Y
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0,            1, 0           ],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    t = np.array([3.0, 0.0, 0.0])  # move B 3m in X (example)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    # apply to B, then merge
    pcd_B.transform(T)
    combined = pcd_A + pcd_B

    o3d.io.write_point_cloud("merged_room.ply", combined)
    np.save("transformation_wallB.npy", T)

    print("Saved merged_room.ply and transformation_wallB.npy")
    return T

# transform cams (listing 2 / complete_workflow)
def transform_camera_pose(R_old, t_old, T):
    R_transform = T[:3, :3]
    t_transform = T[:3, 3]

    C_old = -R_old.T @ t_old.reshape(3, 1)
    R_new = R_transform @ R_old
    C_new = R_transform @ C_old + t_transform.reshape(3, 1)
    t_new = -R_new @ C_new
    return R_new, t_new

def transform_cameras_wallB(T):
    with open("cameras_wallB.json", "r") as f:
        cams = json.load(f)

    for cam in cams["cameras"]:
        R_old = np.array(cam["matrix_R"])
        t_old = np.array(cam["vector_t"])
        R_new, t_new = transform_camera_pose(R_old, t_old, T)
        cam["matrix_R"] = R_new.tolist()
        cam["vector_t"] = t_new.flatten().tolist()

    with open("cameras_wallB_transformed.json", "w") as f:
        json.dump(cams, f, indent=2)

    print("Saved cameras_wallB_transformed.json")
    return "cameras_wallB_transformed.json"

# combine the cams and export for the viewer
def combine_cameras_for_viewer():
    with open("cameras_wallA.json", "r") as f:
        A = json.load(f)
    with open("cameras_wallB_transformed.json", "r") as f:
        B = json.load(f)

    all_cams = {
        "cameras": A["cameras"] + B["cameras"],
        "num_cameras": len(A["cameras"]) + len(B["cameras"])
    }
    with open("viewer_data/cameras.json", "w") as f:
        json.dump(all_cams, f, indent=2)
    print("Wrote merged cameras.json in viewer_data/")

def ply_to_points_json(ply_path="merged_room.ply",
                       output="viewer_data/points.json",
                       max_points=100000):
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]

    data = {"positions": pts.astype(float).ravel().tolist()}
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f)
    print(f"Wrote merged points.json with {len(pts)} points")

if __name__ == "__main__":
    print("Step 1: merge clouds")
    T = merge_and_save()

    print("\nStep 2: transform cameras for Wall B")
    transform_cameras_wallB(T)

    print("\nStep 3: combine cameras and export for viewer")
    combine_cameras_for_viewer()

    print("\nStep 4: convert merged_room.ply to points.json")
    ply_to_points_json()
    print("\nDone. Use viewer_data/index.html with new cameras.json + points.json")
