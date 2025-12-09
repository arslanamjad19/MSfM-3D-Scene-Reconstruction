import json
import os
os.environ["WEBRTC_PORT"] = "8890"
os.environ["WEBRTC_IP"] = "localhost"

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter, defaultdict
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import open3d as o3d

# hyperparams

folder = "Images/test_set"
indices = list(range(1, 34)) # Img1 ... Img11
algo = "SIFT" # or "ORB"
target_w = 3000
output_dir = "Ply_pose2_DLC" # e.g. "Ply_pose1_DLC" or "Ply_pose2_DLC"

min_pnp_pts = 20 # minimum 3D-2D correspondences to attempt PnP # was 12
min_pnp_inliers = 60 # minimum PnP inliers to accept pose (started with 60)
pnp_reproj_err_px = 2.0

# triangulation quality
triang_E_thresh_px = 1.5 # was 2.0
triang_reproj_thresh_px = 2.0 # was 3.0
triang_min_parallax_deg = 2.0 # was 0.5

# when view k fails, try matching against up to N previous posed cameras
max_back_refs = 3

PROJECT_ROOT = Path(".").resolve()

# helper functs
def build_K_from_image(img, f_scale=0.8):
    H, W = img.shape[:2]
    fx = fy = float(W) * float(f_scale)
    cx, cy = W / 2.0, H / 2.0
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def write_ply(path, points):
    pts = np.asarray(points, dtype=np.float32)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in pts:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def get_detector(name="SIFT"):
    name = name.upper()
    if name == "SIFT":
        return cv.SIFT_create(), cv.NORM_L2, 0.75
    if name == "ORB":
        return cv.ORB_create(nfeatures=3000), cv.NORM_HAMMING, 0.78
    raise ValueError("Use \"SIFT\" or \"ORB\"")


def decompose_E_four(E):
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]], dtype=np.float64)

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2].reshape(3, 1)

    if np.linalg.det(R1) < 0:
        R1 *= -1
    if np.linalg.det(R2) < 0:
        R2 *= -1

    return [(R1,  t), (R1, -t), (R2,  t), (R2, -t)]


def triangulate_points_two_views(K, R1, t1, R2, t2, pts1, pts2):
    P1 = K @ np.hstack([R1, t1])
    P2 = K @ np.hstack([R2, t2])
    Xh = cv.triangulatePoints(P1.astype(np.float64),
                              P2.astype(np.float64),
                              pts1.T.astype(np.float64),
                              pts2.T.astype(np.float64))
    X = (Xh[:3] / Xh[3]).T
    return X


def cheirality_mask(R, t, X):
    # depth in cam = (R X + t)[2]
    Z = (X @ R[2, :].T) + t[2, 0]
    return Z > 0


def select_pose_by_cheirality(E, K, pts1, pts2):
    best = None
    R1 = np.eye(3)
    t1 = np.zeros((3, 1))
    for (R2, t2) in decompose_E_four(E):
        X = triangulate_points_two_views(K, R1, t1, R2, t2, pts1, pts2)
        good1 = cheirality_mask(R1, t1, X)
        good2 = cheirality_mask(R2, t2, X)
        good = good1 & good2
        cnt = int(np.count_nonzero(good))
        if best is None or cnt > best[0]:
            best = (cnt, R2, t2, X, good)
    return best


def project_points(K, R, t, X):
    Xh = np.hstack([X, np.ones((len(X), 1), dtype=np.float64)])
    P = K @ np.hstack([R, t])
    p = (P @ Xh.T).T
    z = p[:, 2:3]
    ok = z[:, 0] > 1e-10
    uv = np.zeros((len(X), 2), dtype=np.float64)
    uv[ok] = p[ok, :2] / z[ok]
    uv[~ok] = np.nan
    return uv


def camera_center(R, t):
    return (-R.T @ t).reshape(3)


def parallax_angle_deg(C1, C2, X):
    v1 = X - C1
    v2 = X - C2
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    c = float(np.dot(v1, v2) / (n1 * n2))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))

# load images
imgs = []
for i in indices:
    im = cv.imread(f"{folder}/Img{i}.jpg", cv.IMREAD_COLOR)
    assert im is not None, f"Missing {folder}/Img{i}.jpg"
    if im.shape[1] > target_w:
        s = target_w / im.shape[1]
        im = cv.resize(im, (target_w, int(im.shape[0] * s)), interpolation=cv.INTER_AREA)
    imgs.append(im)

K_seq = build_K_from_image(imgs[0], f_scale=0.8)

# detect once
det, norm, ratio = get_detector(algo)
kps = []
descs = []
for im in imgs:
    g = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    kp, des = det.detectAndCompute(g, None)
    kps.append(kp)
    descs.append(des)

# match helper
_bf = None
def match_ij(i, j):
    global _bf
    if _bf is None:
        _bf = cv.BFMatcher(norm, crossCheck=False)
    if descs[i] is None or descs[j] is None:
        return [], np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    knn = _bf.knnMatch(descs[i], descs[j], k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    pts_i = np.float32([kps[i][m.queryIdx].pt for m in good]) if good else np.zeros((0, 2), dtype=np.float32)
    pts_j = np.float32([kps[j][m.trainIdx].pt for m in good]) if good else np.zeros((0, 2), dtype=np.float32)
    return good, pts_i, pts_j

# current state of pipelein
cams_seq = {}
cams_seq[0] = (np.eye(3), np.zeros((3, 1)))

points_3d = []
pid_of = [dict() for _ in imgs]

# (cam1, cam2) - we're doing 2 cams at a time
g12, p1, p2 = match_ij(0, 1)
if len(g12) < 8:
    raise RuntimeError("Not enough matches to bootstrap.")

E, maskE = cv.findEssentialMat(p1, p2, K_seq, method=cv.RANSAC, prob=0.999, threshold=2.0)
if E is None or maskE is None:
    raise RuntimeError("EssentialMat RANSAC failed in bootstrap.")

maskE = (maskE.ravel() != 0)
idx_E = np.flatnonzero(maskE)

p1_in = p1[maskE]
p2_in = p2[maskE]

cnt, R2, t2, X12_all, mask12 = select_pose_by_cheirality(E, K_seq, p1_in, p2_in)
cams_seq[1] = (R2, t2)

# map cheirality-positive points back to match indices
idx_keep = idx_E[mask12]
X12 = X12_all[mask12]

for match_idx, X in zip(idx_keep.tolist(), X12):
    m = g12[match_idx]
    qi = m.queryIdx
    ti = m.trainIdx
    pid = len(points_3d)
    points_3d.append(X)
    pid_of[0][qi] = pid
    pid_of[1][ti] = pid

print(f"Bootstrapped: cam2 inliers={len(X12)} | map points={len(points_3d)}")

# from phase 2 - initial dataset had 12 images from CAD LAB @ SSE
last_good = 1

def build_2d3d_from_matches(ref, k, good_matches):
    obj3d = []
    img2d = []
    idx_ref = []
    idx_cur = []
    for m in good_matches:
        q = m.queryIdx
        t = m.trainIdx
        if q in pid_of[ref]:
            pid = pid_of[ref][q]
            obj3d.append(points_3d[pid])
            img2d.append(kps[k][t].pt)
            idx_ref.append(q)
            idx_cur.append(t)
    if len(obj3d) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 2), dtype=np.float64), [], []
    return (np.asarray(obj3d, dtype=np.float64),
            np.asarray(img2d, dtype=np.float64),
            idx_ref,
            idx_cur)

for k in range(2, len(imgs)):
    # choose references: last_good, then a couple earlier posed cams
    posed = [i for i in sorted(cams_seq.keys()) if i < k]
    posed.sort(reverse=True)
    refs = []
    if last_good in cams_seq:
        refs.append(last_good)
    for r in posed:
        if r == last_good:
            continue
        refs.append(r)
        if len(refs) >= max_back_refs:
            break

    best = None  # (num_inliers, ref_used, good_matches, idx_ref, idx_cur, Rk, tk, inliers)
    for ref in refs:
        good, pts_ref, pts_cur = match_ij(ref, k)
        if len(good) < 8:
            continue

        obj3d, img2d, idx_ref, idx_cur = build_2d3d_from_matches(ref, k, good)
        if len(obj3d) < min_pnp_pts:
            continue

        success, rvec, tvec, inliers = cv.solvePnPRansac(
            objectPoints=obj3d,
            imagePoints=img2d,
            cameraMatrix=K_seq,
            distCoeffs=None,
            iterationsCount=2000,
            reprojectionError=pnp_reproj_err_px,
            confidence=0.999,
            flags=cv.SOLVEPNP_ITERATIVE
        )
        if not success or inliers is None:
            continue
        ninl = int(len(inliers))
        if best is None or ninl > best[0]:
            Rk, _ = cv.Rodrigues(rvec)
            tk = tvec.reshape(3, 1)
            best = (ninl, ref, good, idx_ref, idx_cur, Rk, tk, inliers)

    if best is None or best[0] < min_pnp_inliers:
        msg = "0" if best is None else str(best[0])
        print(f"[cam{k+1}] PnP failed/too weak (best inliers={msg}). Skipping this view.")
        continue

    ninl, ref, good, idx_ref, idx_cur, Rk, tk, inliers = best
    cams_seq[k] = (Rk, tk)
    last_good = k
    print(f"[cam{k+1}] pose estimated with {ninl} PnP inliers. (ref=cam{ref+1})")

    # propagate pids into current frame from the pnp inliers
    for ii in inliers.ravel().tolist():
        q = idx_ref[ii]
        t = idx_cur[ii]
        if q in pid_of[ref]:
            pid_of[k][t] = pid_of[ref][q]

    # triangulate new points between (ref, k) but only the ones that are inliers post-RANSAC
    R_ref, t_ref = cams_seq[ref]
    pts_ref_all = np.float32([kps[ref][m.queryIdx].pt for m in good]) if good else np.zeros((0, 2), dtype=np.float32)
    pts_cur_all = np.float32([kps[k][m.trainIdx].pt for m in good]) if good else np.zeros((0, 2), dtype=np.float32)

    new_tri = 0
    if len(good) >= 8:
        E_fk, mask_fk = cv.findEssentialMat(
            pts_ref_all, pts_cur_all, K_seq,
            method=cv.RANSAC, prob=0.999, threshold=triang_E_thresh_px
        )
        if E_fk is not None and mask_fk is not None:
            inl_fk = (mask_fk.ravel() != 0)
        else:
            inl_fk = np.zeros((len(good),), dtype=bool)

        C_ref = camera_center(R_ref, t_ref)
        C_k = camera_center(Rk, tk)

        for m, keep in zip(good, inl_fk.tolist()):
            if not keep:
                continue

            qi = m.queryIdx
            ti = m.trainIdx

            # skip if already mapped
            if (qi in pid_of[ref]) or (ti in pid_of[k]):
                continue

            pt_ref = np.float64(kps[ref][qi].pt).reshape(1, 2)
            pt_cur = np.float64(kps[k][ti].pt).reshape(1, 2)

            X = triangulate_points_two_views(K_seq, R_ref, t_ref, Rk, tk, pt_ref, pt_cur)[0]

            if not np.all(np.isfinite(X)):
                continue

            # cheirality in both cams
            z_ref = float(R_ref[2, :].dot(X) + t_ref[2, 0])
            z_k = float(Rk[2, :].dot(X) + tk[2, 0])
            if z_ref <= 0 or z_k <= 0:
                continue

            # parallax gate
            ang = parallax_angle_deg(C_ref, C_k, X)
            if ang < triang_min_parallax_deg:
                continue

            # reprojection gate
            uv1 = project_points(K_seq, R_ref, t_ref, X.reshape(1, 3))[0]
            uv2 = project_points(K_seq, Rk, tk, X.reshape(1, 3))[0]
            if not np.all(np.isfinite(uv1)) or not np.all(np.isfinite(uv2)):
                continue
            e1 = float(np.linalg.norm(uv1 - pt_ref.reshape(2)))
            e2 = float(np.linalg.norm(uv2 - pt_cur.reshape(2)))
            if max(e1, e2) > triang_reproj_thresh_px:
                continue

            pid = len(points_3d)
            points_3d.append(X)
            pid_of[ref][qi] = pid
            pid_of[k][ti] = pid
            new_tri += 1

    print(f"[cam{k+1}] triangulated new points: {new_tri}")

# final cloud + plotting
X_seq = np.asarray(points_3d, dtype=np.float64)
print(f"\nSequential SfM done: cams={len(cams_seq)} / points={len(X_seq)}")

if len(X_seq) > 0:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X_seq[:, 0], X_seq[:, 1], X_seq[:, 2], s=2)
    ax.set_title("12-view sparse point cloud")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

write_ply(PROJECT_ROOT / "sequence_12views_sparse.ply", X_seq)
print("Saved:", PROJECT_ROOT / "sequence_12views_sparse.ply")

# filtering outliers
def per_point_observations(pid_of, kps):
    obs = {}
    for i, mapping in enumerate(pid_of):
        for kp_idx, pid in mapping.items():
            obs.setdefault(pid, []).append((i, kps[i][kp_idx].pt))
    return obs


def reproj_errors_for_point(pid, Xp, obs, cams, K):
    errs = []
    if pid not in obs:
        return errs
    Xh = np.array([Xp[0], Xp[1], Xp[2], 1.0], dtype=np.float64)
    for i, (u, v) in obs[pid]:
        if i not in cams:
            continue
        R, t = cams[i]
        P = K @ np.hstack([R, t])
        p = P @ Xh
        if p[2] <= 1e-8:
            errs.append(1e9)
            continue
        uhat, vhat = p[0] / p[2], p[1] / p[2]
        errs.append(np.hypot(uhat - u, vhat - v))
    return errs


obs = per_point_observations(pid_of, kps)
err_med = np.full(len(X_seq), np.inf, dtype=np.float64)
views = np.zeros(len(X_seq), dtype=np.int32)

for pid in range(len(X_seq)):
    e = reproj_errors_for_point(pid, X_seq[pid], obs, cams_seq, K_seq)
    if len(e) > 0:
        err_med[pid] = float(np.median(e))
        views[pid] = len(e)

rep_mask = (views >= 2) & (err_med <= 2.0)
X_rep = X_seq[rep_mask]
print(f"Reproj filter -> kept {len(X_rep)}/{len(X_seq)} points (>=2 views & median err <= 2 px)")

if len(X_rep) >= 10:
    med = np.median(X_rep, axis=0)
    d = np.linalg.norm(X_rep - med, axis=1)
    hi = np.percentile(d, 97)
    mask_3d = (d <= hi)
else:
    mask_3d = np.ones(len(X_rep), dtype=bool)

X_seq_filtered = X_rep[mask_3d]
print(f"3D trim -> kept {len(X_seq_filtered)} points after robust cutoff")

write_ply(PROJECT_ROOT / "sequence_12views_sparse_raw.ply", X_seq)
write_ply(PROJECT_ROOT / "sequence_12views_sparse_filtered.ply", X_seq_filtered)
print("Saved raw & filtered PLYs.")

# open3D visualisation
try:
    def cam_frame(T=np.eye(4), size=0.2):
        f = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
        f.transform(T)
        return f

    def T_world_from_cam(R, t):
        # world->cam is [R|t], so cam->world is [R^T | -R^T t]
        Twc = np.eye(4)
        Twc[:3, :3] = R.T
        Twc[:3, 3] = (-R.T @ t).ravel()
        return Twc

    if len(X_seq_filtered) >= 5:
        extent = np.percentile(np.linalg.norm(X_seq_filtered - np.median(X_seq_filtered, axis=0), axis=1), 90)
        axis_size = max(0.05, 0.15 * float(extent))
    else:
        axis_size = 0.2

    frames = [cam_frame(np.eye(4), size=axis_size)]
    for i in sorted(cams_seq.keys()):
        R, t = cams_seq[i]
        frames.append(cam_frame(T_world_from_cam(R, t), size=axis_size))

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X_seq_filtered))
    pcd.paint_uniform_color([0.2, 0.6, 1.0])

    try:
        o3d.visualization.draw([pcd, *frames])
    except Exception:
        o3d.visualization.draw_geometries([pcd, *frames])
except Exception as e:
    print("Open3D not available:", e)

# fast BA
cam_ids = sorted(cams_seq.keys())
cam_to_idx = {c: i for i, c in enumerate(cam_ids)}
num_cams = len(cam_ids)

obs_cam = []
obs_pid = []
obs_uv = []
for c in cam_ids:
    mapping = pid_of[c]
    for kp_idx, pid in mapping.items():
        u, v = kps[c][kp_idx].pt
        obs_cam.append(cam_to_idx[c])
        obs_pid.append(pid)
        obs_uv.append([u, v])

obs_cam = np.asarray(obs_cam, dtype=np.int32)
obs_pid = np.asarray(obs_pid, dtype=np.int32)
obs_uv = np.asarray(obs_uv, dtype=np.float64)

count_by_pid = Counter(obs_pid.tolist())
valid = np.array([count_by_pid[p] >= 2 for p in obs_pid], dtype=bool)
obs_cam, obs_pid, obs_uv = obs_cam[valid], obs_pid[valid], obs_uv[valid]


def quick_proj_errs(pids, cams_seq, X, K):
    sel = defaultdict(list)
    for ic, pid, (u, v) in zip(obs_cam, pids, obs_uv):
        if len(sel[pid]) == 0:
            sel[pid] = [ic, u, v]
    errs = {}
    for pid, (ic, u, v) in sel.items():
        R, t = cams_seq[cam_ids[ic]]
        P = K @ np.hstack([R, t])
        Xh = np.array([*X[pid], 1.0])
        p = P @ Xh
        if p[2] <= 1e-8:
            errs[pid] = 1e9
        else:
            errs[pid] = float(np.hypot(p[0] / p[2] - u, p[1] / p[2] - v))
    return errs


err_rank = quick_proj_errs(obs_pid, cams_seq, X_seq, K_seq)

MAX_POINTS = 3000
good_pids = [pid for pid, e in err_rank.items() if e <= 3.0]
views_per_pid = Counter(obs_pid.tolist())
good_pids.sort(key=lambda p: (-views_per_pid[p], err_rank[p]))

sel_pids = set(good_pids[:MAX_POINTS])
mask_sel = np.array([p in sel_pids for p in obs_pid], dtype=bool)
obs_cam = obs_cam[mask_sel]
obs_pid = obs_pid[mask_sel]
obs_uv = obs_uv[mask_sel]

uniq_pids = sorted(set(obs_pid.tolist()))
pid_old_to_new = {p: i for i, p in enumerate(uniq_pids)}
obs_pid = np.array([pid_old_to_new[p] for p in obs_pid], dtype=np.int32)

X0 = X_seq[uniq_pids].astype(np.float64).copy()
M = len(uniq_pids)

print(f"BA set: cams={num_cams}, points={M}, observations={len(obs_cam)}")


def pack_params(cams_seq, X):
    cam_params = []
    for i_c, c in enumerate(cam_ids):
        if i_c == 0:
            continue
        R, t = cams_seq[c]
        rvec, _ = cv.Rodrigues(R)
        cam_params.append(np.hstack([rvec.ravel(), t.ravel()]))
    cam_params = np.concatenate(cam_params) if len(cam_params) else np.zeros(0)
    return cam_params, X.ravel()


def unpack_params(cam_params, X_flat):
    cams = {}
    cams[0] = cams_seq[cam_ids[0]]
    idx = 0
    for i_c, c in enumerate(cam_ids):
        if i_c == 0:
            continue
        r = cam_params[idx:idx + 3]
        t = cam_params[idx + 3:idx + 6]
        idx += 6
        R, _ = cv.Rodrigues(r.reshape(3, 1))
        cams[i_c] = (R, t.reshape(3, 1))
    X = X_flat.reshape(-1, 3)
    return cams, X


m = len(obs_cam) * 2
n = 6 * (num_cams - 1) + 3 * M
J = lil_matrix((m, n), dtype=int)

for kk, (ic, pid) in enumerate(zip(obs_cam, obs_pid)):
    row_u = 2 * kk
    row_v = 2 * kk + 1
    if ic != 0:
        j0 = 6 * (ic - 1)
        J[row_u, j0:j0 + 6] = 1
        J[row_v, j0:j0 + 6] = 1
    jp = 6 * (num_cams - 1) + 3 * pid
    J[row_u, jp:jp + 3] = 1
    J[row_v, jp:jp + 3] = 1

J = J.tocsr()
K = K_seq.copy()


def residuals(theta):
    cam_params = theta[:6 * (num_cams - 1)]
    X_flat = theta[6 * (num_cams - 1):]
    cams_ba, X = unpack_params(cam_params, X_flat)
    res = np.empty((len(obs_cam) * 2,), dtype=np.float64)

    for kk, (ic, pid, (u, v)) in enumerate(zip(obs_cam, obs_pid, obs_uv)):
        R, t = cams_ba.get(ic, cams_seq[cam_ids[ic]])
        Xp = X[pid]
        Xh = np.array([Xp[0], Xp[1], Xp[2], 1.0], dtype=np.float64)
        P = K @ np.hstack([R, t])
        p = P @ Xh
        if p[2] <= 1e-8:
            res[2 * kk] = 1e3
            res[2 * kk + 1] = 1e3
        else:
            uh, vh = p[0] / p[2], p[1] / p[2]
            res[2 * kk] = uh - u
            res[2 * kk + 1] = vh - v
    return res


cam_params0, X0_flat = pack_params(cams_seq, X0)
theta0 = np.concatenate([cam_params0, X0_flat])

print(f"Optimizing variables={theta0.size} with sparse Jacobian...")
res = least_squares(
    residuals, theta0,
    method="trf",
    jac_sparsity=J,
    loss="huber", f_scale=2.0,
    max_nfev=120,
    xtol=1e-8, ftol=1e-8, gtol=1e-8,
    verbose=2
)

cam_params_opt = res.x[:6 * (num_cams - 1)]
Xopt_flat = res.x[6 * (num_cams - 1):]
cams_ba, Xopt = unpack_params(cam_params_opt, Xopt_flat)

for i_c, c in enumerate(cam_ids):
    if i_c == 0:
        continue
    cams_seq[c] = cams_ba[i_c]

X_seq[uniq_pids] = Xopt
print("Fast BA done. cams_seq and X_seq updated.")

# post-BA clean-up and saving
def per_point_obs(pid_of, kps):
    obs2 = {}
    for i, mapping in enumerate(pid_of):
        for kp_idx, pid in mapping.items():
            obs2.setdefault(pid, []).append((i, kps[i][kp_idx].pt))
    return obs2


def point_reproj_errs(pid, Xp, obs2, cams, K):
    if pid not in obs2:
        return []
    errs = []
    Xh = np.array([Xp[0], Xp[1], Xp[2], 1.0], dtype=np.float64)
    for i, (u, v) in obs2[pid]:
        if i not in cams:
            continue
        R, t = cams[i]
        P = K @ np.hstack([R, t])
        p = P @ Xh
        if p[2] <= 1e-8:
            errs.append(np.nan)
            continue
        uh, vh = p[0] / p[2], p[1] / p[2]
        errs.append(float(np.hypot(uh - u, vh - v)))
    return errs


obs2 = per_point_obs(pid_of, kps)
err_med = np.full(len(X_seq), np.nan)
views = np.zeros(len(X_seq), dtype=int)

for pid in range(len(X_seq)):
    e = np.array(point_reproj_errs(pid, X_seq[pid], obs2, cams_seq, K_seq), dtype=float)
    if e.size:
        views[pid] = np.isfinite(e).sum()
        if np.isfinite(e).any():
            err_med[pid] = np.nanmedian(e)

valid = np.isfinite(err_med)
vals = err_med[valid]
print(f"Points with valid repro: {valid.sum()}/{len(X_seq)}")
if vals.size:
    print(f"Median(err) px: mean(trimmed)={np.mean(vals[vals < 10]):.2f}, "
          f"median={np.median(vals):.2f}, p90={np.percentile(vals, 90):.2f}, "
          f"p95={np.percentile(vals, 95):.2f}")

# remove outliers
mask_rep = (views >= 2) & (np.isfinite(err_med)) & (err_med <= 2.0)
X_rep = X_seq[mask_rep]

if len(X_rep) >= 10:
    med = np.median(X_rep, axis=0)
    d = np.linalg.norm(X_rep - med, axis=1)
    hi = np.percentile(d, 97)
    mask_3d = (d <= hi)
else:
    mask_3d = np.ones(len(X_rep), dtype=bool)

X_seq_filtered = X_rep[mask_3d]
print(f"Kept {len(X_seq_filtered)} / {len(X_seq)} points after pruning.")

write_ply(PROJECT_ROOT / "sequence_12views_sparse_raw.ply", X_seq)
write_ply(PROJECT_ROOT / "sequence_12views_sparse_filtered.ply", X_seq_filtered)
print("Saved PLYs (raw & filtered).")

# export to phase 3
from phase3_export import export_all_for_phase3

if len(cams_seq) > 0 and len(X_seq_filtered) > 0:
    export_all_for_phase3(
        cams_seq=cams_seq,
        X_seq=X_seq_filtered,
        pid_of=pid_of,
        kps=kps,
        indices=indices,
        folder=folder,
        output_dir=output_dir
    )
else:
    print("WARNING: Not enough data for Phase 3 export (cams or points).")
