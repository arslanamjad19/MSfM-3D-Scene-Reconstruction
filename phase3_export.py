import json
import os
import numpy as np
import cv2 as cv
from pathlib import Path

PROJECT_ROOT = Path(".").resolve()

# math functions

def _rotmat_to_quat_xyzw(R):
    """
    Convert 3x3 rotation matrix to quaternion (x, y, z, w).
    Stable-ish implementation for proper rotation matrices.
    """
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

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# exports

def export_cameras_to_json(cams_seq, output_path, indices=None):
    """
    Export camera centers + a Three.js-friendly quaternion.
    """
    cameras_data = []

    # converts openCV camera coords to 3 camera coords: (x, y, z) -> (x, -y, -z)
    S = np.diag([1.0, -1.0, -1.0]).astype(np.float64)

    for cam_idx in sorted(cams_seq.keys()):
        R_wc, t_wc = cams_seq[cam_idx]  # (R, t) world->cam
        R_wc = np.asarray(R_wc, dtype=np.float64)
        t_wc = np.asarray(t_wc, dtype=np.float64).reshape(3, 1)

        # camera center in world
        C = (-R_wc.T @ t_wc).reshape(3)

        # camera-to-world rotation (opencv): R_cw = R_wc^T
        R_cw = R_wc.T

        # convert to 3js camera convention in camera-local axes
        R_three = R_cw @ S
        q = _rotmat_to_quat_xyzw(R_three)

        img_id = None
        if indices is not None and cam_idx < len(indices):
            img_id = int(indices[cam_idx])

        cameras_data.append({
            "id": int(cam_idx),
            "image_id": img_id if img_id is not None else int(cam_idx),
            "image": f"images/img{img_id if img_id is not None else int(cam_idx)}.jpeg",
            "position": {"x": float(C[0]), "y": float(C[1]), "z": float(C[2])},
            "quaternion": {"x": float(q[0]), "y": float(q[1]), "z": float(q[2]), "w": float(q[3])},

            "matrix_R_wc": R_wc.tolist(),
            "vector_t_wc": t_wc.reshape(3).tolist()
        })

    _ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"cameras": cameras_data, "num_cameras": len(cameras_data)}, f, indent=2)

    print(f"Exported {len(cameras_data)} cameras to {output_path}")
    return output_path


def export_point_cloud_to_json(X, output_path, max_points=10000):
    X = np.asarray(X, dtype=np.float64)
    if len(X) > max_points:
        sel = np.random.choice(len(X), max_points, replace=False)
        X = X[sel]

    points_data = {"positions": X.flatten().tolist(), "count": int(len(X))}
    _ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(points_data, f, indent=2)

    print(f"Exported {len(X)} points to {output_path}")
    return output_path


def compute_view_graph(cams_seq, X_seq, pid_of, kps, min_shared_points=50):
    view_graph = {int(cam_idx): [] for cam_idx in cams_seq.keys()}
    cam_indices = sorted([int(k) for k in cams_seq.keys()])

    for i, cam_i in enumerate(cam_indices):
        for cam_j in cam_indices[i + 1:]:
            if cam_i >= len(pid_of) or cam_j >= len(pid_of):
                continue
            points_i = set(pid_of[cam_i].values()) if isinstance(pid_of[cam_i], dict) else set()
            points_j = set(pid_of[cam_j].values()) if isinstance(pid_of[cam_j], dict) else set()
            shared = points_i & points_j
            if len(shared) >= min_shared_points:
                view_graph[cam_i].append({"target": cam_j, "shared_points": int(len(shared))})
                view_graph[cam_j].append({"target": cam_i, "shared_points": int(len(shared))})

    for cam_idx in view_graph:
        view_graph[cam_idx].sort(key=lambda x: x["shared_points"], reverse=True)

    return view_graph


def export_view_graph_to_json(view_graph, output_path):
    _ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(view_graph, f, indent=2)
    print(f"Exported view graph to {output_path}")
    return output_path


def copy_images_to_viewer(indices, folder, output_folder):
    """
    Robustly copy Img{idx}.{jpg|jpeg|png} -> img{idx}.jpeg (and resize for web).
    """
    _ensure_dir(output_folder)

    exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    copied = 0

    for i in indices:
        src_path = None
        for ext in exts:
            p = os.path.join(folder, f"Img{i}{ext}")
            if os.path.exists(p):
                src_path = p
                break
        if src_path is None:
            continue

        img = cv.imread(src_path)
        if img is None:
            continue

        if img.shape[1] > 1920:
            scale = 1920.0 / img.shape[1]
            img = cv.resize(img, (1920, int(img.shape[0] * scale)), interpolation=cv.INTER_AREA)

        dst = os.path.join(output_folder, f"img{i}.jpeg")
        cv.imwrite(dst, img)
        copied += 1

    print(f"Copied {copied}/{len(indices)} images into {output_folder}")


def create_html_viewer(output_path):
    """
    Phase-3 style viewer:
    - shows current image full-screen
    - click chooses a neighbor from view_graph
    - smooth lerp+slerp camera motion + cross-fade images
    - point cloud visible during transition (and optional toggle)
    """
    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Photosynth-Style Viewer</title>
  <style>
    html, body { margin:0; padding:0; overflow:hidden; background:#000; }
    #bg {
      position:fixed; inset:0; overflow:hidden;
      pointer-events:none;
    }
    #bg img {
      position:absolute; inset:0;
      width:100%; height:100%;
      object-fit:contain;
      opacity:0;
      transition: opacity 0.8s ease;
      background:#000;
    }
    #bg img.show { opacity:1; }

    #canvas { position:fixed; inset:0; }

    .ui {
      position:fixed; left:18px; top:18px;
      background:rgba(0,0,0,0.70);
      color:#fff; font-family:Arial, sans-serif;
      padding:14px 14px; border-radius:12px;
      min-width:240px;
      box-shadow:0 10px 30px rgba(0,0,0,0.35);
      z-index:10;
    }
    .ui h3 { margin:0 0 6px 0; font-size:16px; }
    .row { display:flex; gap:8px; margin-top:10px; flex-wrap:wrap; }
    button {
      border:0; border-radius:10px;
      padding:9px 12px; cursor:pointer;
      background:#2f6fed; color:white;
      font-weight:600;
    }
    button.secondary { background:#2f2f2f; }
    label { font-size:13px; opacity:0.9; user-select:none; }
    .muted { font-size:12px; opacity:0.75; margin-top:6px; }
  </style>
</head>
<body>
  <div id="bg">
    <img id="imgA" />
    <img id="imgB" />
  </div>
  <canvas id="canvas"></canvas>

  <div class="ui">
    <h3>Virtual Tour Controls</h3>
    <div>Camera: <span id="camLabel">-</span></div>

    <div class="row">
      <button id="prevBtn" class="secondary">← Prev</button>
      <button id="nextBtn">Next →</button>
    </div>

    <div class="row">
      <button id="resetBtn" class="secondary">Reset</button>
    </div>

    <div class="row">
      <label><input type="checkbox" id="showPts" checked /> Show point cloud</label>
    </div>

    <div class="muted">
      Tip: click on the view to jump to the “best” neighbor using the view graph.
    </div>
  </div>

<script type="module">
  import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.module.js";

  const canvas = document.getElementById("canvas");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias:true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);

  const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.01, 1e6);

  // lighting only matters for the little camera cones
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const d = new THREE.DirectionalLight(0xffffff, 0.6);
  d.position.set(5, 5, 5);
  scene.add(d);

  let cameraData = [];
  let viewGraph = {};
  let pointCloud = null;

  let current = 0;
  let transitioning = false;
  let t0 = 0;
  const duration = 900; // ms

  let startPos, endPos;
  let startQuat, endQuat;

  const imgA = document.getElementById("imgA");
  const imgB = document.getElementById("imgB");
  let frontImg = imgA;
  let backImg = imgB;

  function setBgImage(path) {
    // swap front/back and fade
    backImg.src = path;
    backImg.onload = () => {
      backImg.classList.add("show");
      frontImg.classList.remove("show");
      const tmp = frontImg;
      frontImg = backImg;
      backImg = tmp;
    };
  }

  function applyCamera(i) {
    const c = cameraData[i];
    camera.position.set(c.position.x, c.position.y, c.position.z);

    if (c.quaternion) {
      camera.quaternion.set(c.quaternion.x, c.quaternion.y, c.quaternion.z, c.quaternion.w);
    }

    document.getElementById("camLabel").textContent = `${i+1} / ${cameraData.length}`;
    setBgImage(c.image || `images/img${c.image_id}.jpeg`);
  }

  function navigateTo(i) {
    if (transitioning || i === current) return;
    const a = cameraData[current];
    const b = cameraData[i];

    startPos = new THREE.Vector3(a.position.x, a.position.y, a.position.z);
    endPos   = new THREE.Vector3(b.position.x, b.position.y, b.position.z);

    startQuat = new THREE.Quaternion(a.quaternion.x, a.quaternion.y, a.quaternion.z, a.quaternion.w);
    endQuat   = new THREE.Quaternion(b.quaternion.x, b.quaternion.y, b.quaternion.z, b.quaternion.w);

    // cross-fade *now* so it overlaps motion
    setBgImage(b.image || `images/img${b.image_id}.jpeg`);

    t0 = performance.now();
    transitioning = true;
    current = i;

    document.getElementById("camLabel").textContent = `${current+1} / ${cameraData.length}`;
  }

  function bestNeighborFromClick(ev) {
    const neighbors = viewGraph[String(current)] || viewGraph[current] || [];
    if (neighbors.length === 0) return null;

    const rect = canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((ev.clientX - rect.left) / rect.width) * 2 - 1,
      -(((ev.clientY - rect.top) / rect.height) * 2 - 1)
    );

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(ndc, camera);
    const dir = raycaster.ray.direction.clone().normalize();

    // choose neighbor whose direction best matches click ray
    const curPos = new THREE.Vector3(
      cameraData[current].position.x,
      cameraData[current].position.y,
      cameraData[current].position.z
    );

    let best = null;
    let bestScore = -1e9;

    for (const e of neighbors) {
      const j = e.target;
      const p = new THREE.Vector3(
        cameraData[j].position.x,
        cameraData[j].position.y,
        cameraData[j].position.z
      );
      const v = p.sub(curPos).normalize();
      const score = dir.dot(v) + 0.0005 * (e.shared_points || 0);
      if (score > bestScore) {
        bestScore = score;
        best = j;
      }
    }

    return best;
  }

  function animate() {
    requestAnimationFrame(animate);

    if (transitioning) {
      const t = (performance.now() - t0) / duration;
      const u = Math.min(1, Math.max(0, t));

      const pos = startPos.clone().lerp(endPos, u);
      const quat = startQuat.clone().slerp(endQuat, u);

      camera.position.copy(pos);
      camera.quaternion.copy(quat);

      if (u >= 1) {
        transitioning = false;
        // snap final state (and ensure correct bg)
        applyCamera(current);
      }
    }

    const showPts = document.getElementById("showPts").checked;
    if (pointCloud) pointCloud.visible = showPts || transitioning;

    renderer.render(scene, camera);
  }

  function addPointCloud(positions) {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geom.computeBoundingBox();

    const bbox = geom.boundingBox;
    const diag = bbox.max.clone().sub(bbox.min).length();
    const size = Math.max(0.001, diag / 800.0);

    const mat = new THREE.PointsMaterial({ size, sizeAttenuation:true, color:0x7fd3ff });
    pointCloud = new THREE.Points(geom, mat);
    scene.add(pointCloud);

    // tighten near/far based on point cloud size
    camera.near = Math.max(0.001, diag / 10000.0);
    camera.far  = Math.max(10.0, diag * 50.0);
    camera.updateProjectionMatrix();
  }

  async function loadData() {
    const cams = await fetch("cameras.json").then(r => r.json());
    const pts  = await fetch("points.json").then(r => r.json());
    viewGraph  = await fetch("view_graph.json").then(r => r.json());

    cameraData = cams.cameras;

    addPointCloud(pts.positions);

    // sanity: if quaternions missing (older exports), bail with message
    for (const c of cameraData) {
      if (!c.quaternion) {
        console.warn("Camera quaternion missing. Re-export using updated phase3_export.py.");
      }
    }

    applyCamera(0);
  }

  // UI
  document.getElementById("nextBtn").onclick = () => navigateTo((current + 1) % cameraData.length);
  document.getElementById("prevBtn").onclick = () => navigateTo((current - 1 + cameraData.length) % cameraData.length);
  document.getElementById("resetBtn").onclick = () => { current = 0; applyCamera(0); };

  canvas.addEventListener("click", (ev) => {
    if (transitioning) return;
    const nb = bestNeighborFromClick(ev);
    if (nb !== null && nb !== undefined) navigateTo(nb);
  });

  window.addEventListener("resize", () => {
    renderer.setSize(window.innerWidth, window.innerHeight);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
  });

  loadData();
  animate();
</script>
</body>
</html>
"""
    _ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Created HTML viewer at {output_path}")
    return output_path


def export_all_for_phase3(cams_seq, X_seq, pid_of, kps, indices, folder, output_dir="viewer_data"):
    print("\n" + "=" * 60)
    print("EXPORTING DATA FOR PHASE 3 VIRTUAL TOUR VIEWER")
    print("=" * 60)

    if not cams_seq or len(X_seq) == 0:
        print("ERROR: No cameras or points to export!")
        return

    _ensure_dir(output_dir)

    cam_path = os.path.join(output_dir, "cameras.json")
    export_cameras_to_json(cams_seq, cam_path, indices=indices)

    points_path = os.path.join(output_dir, "points.json")
    export_point_cloud_to_json(X_seq, points_path, max_points=10000)

    view_graph = compute_view_graph(cams_seq, X_seq, pid_of, kps)
    graph_path = os.path.join(output_dir, "view_graph.json")
    export_view_graph_to_json(view_graph, graph_path)

    copy_images_to_viewer(indices, folder, os.path.join(output_dir, "images"))

    html_path = os.path.join(output_dir, "index.html")
    create_html_viewer(html_path)

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE!")
    print("=" * 60)
    print(f"Open: cd {output_dir} && python -m http.server 8000")
    print("Then: http://localhost:8000/index.html")
    print("=" * 60)
