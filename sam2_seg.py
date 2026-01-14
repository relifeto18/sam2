#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RGB+Cloud → SAM2 masks (2 slots) → per-mask 3D clouds.
Adds random action sampling around tumor centroid in CAMERA FRAME:
  - start (green):   y in [cy+0.02,  cy+0.03]
  - goal  (red):     y in [cy-0.01,  cy+0.00]
  - preshape (orange): y in [cy+0.008, cy+0.010]
Each uses x in [cx-x_half, cx+x_half], z in [cz+z_lo, cz+z_hi].

Hotkeys:
  1/2 = active slot
  r   = run SAM2 on active slot (ROI+clicks optional)
  a   = commit mask → 3D cloud
  v   = view 3D clouds
  s   = sample & visualize ONE start/goal/preshape point
  b   = sample & visualize BATCH (n each)
  g   = grab new RGB+Cloud
  c/C = clear slot/all
  u   = undo last click
  q   = quit
"""

import os, sys, time, argparse, threading
import numpy as np
import cv2
import open3d as o3d
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, PointField
import torch

# ---------- SAM2 ----------
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ===================== DEFAULTS =====================
COLOR_TOPIC = "/camera/realsense2_camera_node/color/image_rect_raw"
CLOUD_TOPIC = "/camera/realsense2_camera_node/depth/color/points"
SLOP_SEC    = 0.20   # allowed timestamp mismatch (sec)
TIMEOUT_SEC = 10.0   # max wait for RGB+cloud

SAM2_CKPT = "/home/desser/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"

MIN_DEPTH = 0.05
MAX_DEPTH = 2.00
ALPHA = 0.45  # overlay transparency
HELPBAR_H = 32
FOOTER_H  = 30
WIN_NAME  = "SAM2 ROI+Clicks"

# ===================== PointCloud2 → structured array =====================
def _fields_to_dtype(fields, point_step, is_bigendian):
    np_types = {
        PointField.INT8:   np.int8,   PointField.UINT8:  np.uint8,
        PointField.INT16:  np.int16,  PointField.UINT16: np.uint16,
        PointField.INT32:  np.int32,  PointField.UINT32: np.uint32,
        PointField.FLOAT32: np.float32, PointField.FLOAT64: np.float64,
    }
    dtype_list, offset = [], 0
    for f in sorted(fields, key=lambda f: f.offset):
        if offset < f.offset:
            dtype_list.append((f'__pad_{offset}', np.uint8, f.offset - offset))
            offset = f.offset
        base = np_types.get(f.datatype)
        if base is None: raise ValueError(f"Unsupported PointField datatype: {f.datatype}")
        count = f.count or 1
        dtype_list.append((f.name, base) if count == 1 else (f.name, base, (count,)))
        offset += np.dtype(base).itemsize * count
    if offset < point_step:
        dtype_list.append((f'__pad_{offset}', np.uint8, point_step - offset))
    dt = np.dtype(dtype_list)
    return dt.newbyteorder('>' if is_bigendian else '<')

def pointcloud2_to_structured_array(msg: PointCloud2):
    if msg.height == 0 or msg.width == 0:
        raise ValueError("Unorganized cloud. Enable 'pointcloud.ordered_pc=True' in RealSense params.")
    dt = _fields_to_dtype(msg.fields, msg.point_step, msg.is_bigendian)
    data = np.frombuffer(msg.data, dtype=dt, count=msg.width * msg.height)
    if msg.is_bigendian != (sys.byteorder == 'big'):
        data = data.byteswap().newbyteorder()
    return data.reshape((msg.height, msg.width))

def stamp_to_float(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9

# ===================== Pair grabber (near-synced) =====================
class PairGrabber(Node):
    def __init__(self, color_topic, cloud_topic, slop_sec, timeout_sec):
        super().__init__('sam2_pair')
        self.bridge = CvBridge()
        self.color_msg = None
        self.cloud_msg = None
        self.color_ts = None
        self.cloud_ts = None
        self.slop = slop_sec
        self.done_evt = threading.Event()
        self.timeout_sec = timeout_sec
        self.create_subscription(Image, color_topic, self._on_color, 10)
        self.create_subscription(PointCloud2, cloud_topic, self._on_cloud, 10)

    def _on_color(self, msg: Image):
        self.color_msg = msg
        self.color_ts = stamp_to_float(msg.header.stamp)
        self._maybe_ready()

    def _on_cloud(self, msg: PointCloud2):
        self.cloud_msg = msg
        self.cloud_ts = stamp_to_float(msg.header.stamp)
        self._maybe_ready()

    def _maybe_ready(self):
        if self.color_ts is None or self.cloud_ts is None:
            return
        if abs(self.color_ts - self.cloud_ts) <= self.slop:
            self.done_evt.set()

    def wait_for_pair(self):
        start = time.time()
        while rclpy.ok() and (time.time() - start) < self.timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.done_evt.is_set():
                return True
        return False

def grab_pair(color_topic, cloud_topic, slop, timeout):
    rclpy.init()
    node = PairGrabber(color_topic, cloud_topic, slop, timeout)
    node.get_logger().info(f"Waiting RGB+Cloud (slop={slop}s):\n  color: {color_topic}\n  cloud: {cloud_topic}")
    ok = node.wait_for_pair()
    if not ok:
        node.get_logger().error("Timed out.")
        rclpy.shutdown()
        return None, None
    try:
        bgr = node.bridge.imgmsg_to_cv2(node.color_msg, desired_encoding='bgr8')
    except Exception as e:
        node.get_logger().error(f"cv_bridge failed: {e}"); rclpy.shutdown(); return None, None
    try:
        arr = pointcloud2_to_structured_array(node.cloud_msg)
    except Exception as e:
        node.get_logger().error(f"PointCloud2 parse failed: {e}"); rclpy.shutdown(); return None, None
    rclpy.shutdown()
    return bgr, arr

# ===================== UI helpers =====================
def overlay_mask(bgr, mask, color_bgr=(0, 200, 255), alpha=ALPHA):
    if mask is None: return bgr
    seg = (mask > 0.5)
    if not np.any(seg): return bgr
    out = bgr.copy()
    out[seg] = ((1 - alpha) * out[seg] + alpha * np.array(color_bgr, dtype=np.uint8)).astype(np.uint8)
    seg_u8 = (seg.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(seg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts: cv2.drawContours(out, cnts, -1, color_bgr, 2)
    return out

def crop_rect(x0, y0, x1, y1, W, H):
    x0, y0 = max(0, min(W-1, x0)), max(0, min(H-1, y0))
    x1, y1 = max(0, min(W,   x1)), max(0, min(H,   y1))
    if x1 <= x0 or y1 <= y0: return None
    return x0, y0, x1, y1

def draw_status(img, text, y=20):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1, cv2.LINE_AA)

# ===================== Mask → 3D lifting =====================
def pcd_from_mask(seg_full_bool, arr_cloud, min_depth, max_depth, flip_yz=True, color_rgb=(1.0, 0.5, 0.0)):
    Hp, Wp = arr_cloud.shape[:2]   # point cloud
    Hc, Wc = seg_full_bool.shape   # segmentation mask
    xo0, yo0 = 0, 0
    xo1, yo1 = min(Wc, Wp), min(Hc, Hp)
    if xo1 <= xo0 or yo1 <= yo0: return None, 0

    seg_crop = seg_full_bool[yo0:yo1, xo0:xo1]
    xyz_roi = np.stack([arr_cloud['x'][yo0:yo1, xo0:xo1],
                        arr_cloud['y'][yo0:yo1, xo0:xo1],
                        arr_cloud['z'][yo0:yo1, xo0:xo1]], axis=-1)   # (H,W,3)
    z = xyz_roi[..., 2]
    finite = np.isfinite(xyz_roi).all(axis=-1)
    depth_ok = (z > min_depth) & (z < max_depth)
    valid = seg_crop & finite & depth_ok
    if not valid.any(): return None, 0

    pts = xyz_roi[valid].reshape(-1, 3).astype(np.float64)

    # Visualization flip (Open3D-friendly) — DO NOT USE for math/extrinsics
    if flip_yz:
        pts[:, 1] *= -1.0
        pts[:, 2] *= -1.0

    cols = np.tile(np.array(color_rgb, dtype=np.float64)[None, :], (pts.shape[0], 1))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd, pts.shape[0]

# ===================== Random action sampling (camera frame) =====================
def sample_actions_around_tumor_cam(
    tumor_pts_cam: np.ndarray,
    x_half=0.001,                  # X in [cx - x_half, cx + x_half]
    # tight Y band (relative to cy)
    y_lo=-0.002, y_hi=-0.001,      # Y in [cy + y_lo, cy + y_hi]
    # Z per action (relative to cz)
    z_start_lo=-0.008, z_start_hi=-0.007,
    z_goal_lo=  0.003, z_goal_hi=  0.004,
    z_preshape_lo=-0.012, z_preshape_hi=-0.011,
    rng: np.random.Generator = None
):
    assert tumor_pts_cam.size > 0, "Need tumor points (mask2) to define center."
    rng = rng or np.random.default_rng()
    cx, cy, cz = tumor_pts_cam.mean(axis=0)

    def rx(): return rng.uniform(cx - x_half, cx + x_half)
    def ry(): return rng.uniform(cy + y_lo,   cy + y_hi)

    xs, ys = rx(), ry()
    zs = rng.uniform(cz + z_start_lo,   cz + z_start_hi)
    start_cam = np.array([xs, ys, zs], dtype=float)

    xg, yg = rx(), ry()
    zg = rng.uniform(cz + z_goal_lo,    cz + z_goal_hi)
    goal_cam  = np.array([xg, yg, zg], dtype=float)

    xp, yp = rx(), ry()
    zp = rng.uniform(cz + z_preshape_lo, cz + z_preshape_hi)
    preshape_cam = np.array([xp, yp, zp], dtype=float)

    return start_cam, goal_cam, preshape_cam

def sample_actions_batch_cam(
    tumor_pts_cam: np.ndarray,
    n: int = 100,
    x_half=0.001,
    y_lo=-0.002, y_hi=-0.001,
    z_start_lo=-0.008, z_start_hi=-0.007,
    z_goal_lo=  0.003, z_goal_hi=  0.004,
    z_preshape_lo=-0.012, z_preshape_hi=-0.011,
    rng: np.random.Generator = None
):
    assert tumor_pts_cam.size > 0, "Need tumor points (mask2) to define center."
    rng = rng or np.random.default_rng()
    cx, cy, cz = tumor_pts_cam.mean(axis=0)

    xs_s = rng.uniform(cx - x_half, cx + x_half, size=n)
    xs_g = rng.uniform(cx - x_half, cx + x_half, size=n)
    xs_p = rng.uniform(cx - x_half, cx + x_half, size=n)

    ys_s = rng.uniform(cy + y_lo,   cy + y_hi,   size=n)
    ys_g = rng.uniform(cy + y_lo,   cy + y_hi,   size=n)
    ys_p = rng.uniform(cy + y_lo,   cy + y_hi,   size=n)

    zs_s = rng.uniform(cz + z_start_lo,   cz + z_start_hi,   size=n)
    zs_g = rng.uniform(cz + z_goal_lo,    cz + z_goal_hi,    size=n)
    zs_p = rng.uniform(cz + z_preshape_lo, cz + z_preshape_hi, size=n)

    start = np.stack([xs_s, ys_s, zs_s], axis=1)
    goal  = np.stack([xs_g, ys_g, zs_g], axis=1)
    pre   = np.stack([xs_p, ys_p, zs_p], axis=1)
    return start, goal, pre

def cam_to_view_flip_for_o3d(p: np.ndarray) -> np.ndarray:
    """Mirror (y,z) to match viewer if clouds were created with flip_yz=True."""
    q = p.copy()
    q[:, 1] *= -1.0
    q[:, 2] *= -1.0
    return q

def align_view_to_cam(vis, viz_flip: bool):
    ctr = vis.get_view_control()
    if viz_flip:
        # you flipped (y,z) → Open3D coords: +x right, +y up, +z into screen
        front = [0, 0, -1]   # look along -Z so +Z comes out of screen
        up    = [0, 1,  0]   # +Y up
    else:
        # raw camera frame: +x right, +y down, +z forward (out of camera)
        # make screen up ≈ -Y, and look along +Z
        front = [0, 0,  1]
        up    = [0,-1,  0]
    params = ctr.convert_to_pinhole_camera_parameters()
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_lookat(params.extrinsic[:3, 3])  # keep current center
    ctr.set_zoom(0.7)

def show_with_cam_align(geoms, viz_flip: bool):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Start (B), Goal (R), Preshape (O)", width=1280, height=720)
    for g in geoms:
        vis.add_geometry(g)
    align_view_to_cam(vis, viz_flip)
    vis.run()
    vis.destroy_window()

def make_sphere_at(p: np.ndarray, radius=0.004, rgb=(1.0, 0.2, 0.2)):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.paint_uniform_color(rgb)
    s.translate(p.astype(float))
    return s

def make_spheres(points: np.ndarray, radius=0.003, rgb=(1.0, 0.2, 0.2)):
    """points: (N,3) → list of small spheres"""
    return [make_sphere_at(points[i], radius=radius, rgb=rgb) for i in range(points.shape[0])]

# ===================== Main =====================
def main():
    ap = argparse.ArgumentParser("RGB+Cloud → SAM2 masks → 3D clouds (2 objects).")
    ap.add_argument('--color', default=COLOR_TOPIC)
    ap.add_argument('--cloud', default=CLOUD_TOPIC)
    ap.add_argument('--slop', type=float, default=SLOP_SEC)
    ap.add_argument('--timeout', type=float, default=TIMEOUT_SEC)
    ap.add_argument('--min_depth', type=float, default=MIN_DEPTH)
    ap.add_argument('--max_depth', type=float, default=MAX_DEPTH)
    ap.add_argument('--checkpoint', type=str, default=SAM2_CKPT)
    ap.add_argument('--config',     type=str, default=SAM2_CFG)
    ap.add_argument('--bfloat16',   action='store_true', help="Use autocast(bfloat16) if supported.")
    ap.add_argument('--viz_flip',   action='store_true',
                    help="Flip Y/Z for visualization only (Open3D). Sampling stays in camera frame.")
    # action sampling box params (meters)
    ap.add_argument('--x_half', type=float, default=0.0)
    ap.add_argument('--y_lo',   type=float, default=0.0)
    ap.add_argument('--y_hi',   type=float, default=0.0)
    ap.add_argument('--z_start_lo',   type=float, default=-0.006)
    ap.add_argument('--z_start_hi',   type=float, default=-0.006)
    ap.add_argument('--z_goal_lo',    type=float, default= 0.005)
    ap.add_argument('--z_goal_hi',    type=float, default= 0.005)
    ap.add_argument('--z_preshape_lo',type=float, default=-0.012)
    ap.add_argument('--z_preshape_hi',type=float, default=-0.012)
    ap.add_argument('--n_samples', type=int, default=100, help="Batch size for 'b' sampling")
    args = ap.parse_args()

    # Grab first pair
    bgr, arr = grab_pair(args.color, args.cloud, args.slop, args.timeout)
    if bgr is None or arr is None: sys.exit(1)
    Hc, Wc = bgr.shape[:2]; Hp, Wp = arr.shape[:2]
    if (Hc, Wc) != (Hp, Wp):
        print(f"[warn] Image {Wc}x{Hc} != cloud {Wp}x{Hp}; using overlap.")

    # Build SAM2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SAM2] device: {device}")
    model = build_sam2(args.config, args.checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)

    # UI state
    roi = {"drag": False, "x0": 0, "y0": 0, "x1": 0, "y1": 0}
    pts_xy_groups = [[], []]      # per slot: [(x,y), ...]
    labels_groups = [[], []]      # per slot: [1/0, ...]
    active_gid = 0                # 0 or 1
    masks = [None, None]          # committed masks per slot
    pcds  = [None, None]          # 3D clouds per slot

    # Compose a frame with help+footer and realize the window BEFORE binding callback
    helpbar = np.zeros((HELPBAR_H, Wc, 3), dtype=np.uint8)
    draw_status(helpbar, "Left: FG | Right: BG | Mid drag: ROI | 1/2: slot | r: run | a: add | u: undo | c/C: clear | v: view3D | s: sample | b: batch | g: grab | q: quit", y=22)
    foot = np.zeros((FOOTER_H, Wc, 3), dtype=np.uint8)
    stack = np.vstack([helpbar, bgr, foot])

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, max(900, min(1600, Wc)), max(700, min(1000, Hc + HELPBAR_H + FOOTER_H)))
    cv2.imshow(WIN_NAME, stack)
    cv2.waitKey(1)

    # Mouse callback
    def on_mouse(event, x, y, flags, _):
        nonlocal roi
        if not (HELPBAR_H <= y < HELPBAR_H + Hc):
            return
        yi = y - HELPBAR_H
        if event == cv2.EVENT_LBUTTONDOWN:
            pts_xy_groups[active_gid].append((x, yi)); labels_groups[active_gid].append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            pts_xy_groups[active_gid].append((x, yi)); labels_groups[active_gid].append(0)
        elif event == cv2.EVENT_MBUTTONDOWN:
            roi.update({"drag": True, "x0": x, "y0": yi, "x1": x, "y1": yi})
        elif event == cv2.EVENT_MOUSEMOVE and roi["drag"]:
            roi["x1"], roi["y1"] = x, yi
        elif event == cv2.EVENT_MBUTTONUP:
            roi["drag"] = False; roi["x1"], roi["y1"] = x, yi

    for _ in range(50):
        try:
            cv2.setMouseCallback(WIN_NAME, on_mouse)
            break
        except cv2.error:
            cv2.waitKey(20)
    else:
        raise RuntimeError("Failed to set mouse callback: Qt window not ready")

    # Colors for two slots
    slot_cols = [(40,220,40), (255,170,40)]  # BGR for points/overlays

    while True:
        vis = bgr.copy()

        # Draw committed masks
        for gid, m in enumerate(masks):
            if m is not None:
                vis = overlay_mask(vis, m, color_bgr=slot_cols[gid], alpha=0.30)

        # ROI rectangle (if any)
        rc = crop_rect(roi["x0"], roi["y0"], roi["x1"], roi["y1"], Wc, Hc)
        if rc is not None:
            x0,y0,x1,y1 = rc
            cv2.rectangle(vis, (x0,y0), (x1,y1), (0,255,255), 2)

        # Draw points for both slots
        for gid, (pts, labs) in enumerate(zip(pts_xy_groups, labels_groups)):
            col = slot_cols[gid]
            for (x,y), lab in zip(pts, labs):
                p = (int(x), int(y))
                if lab == 1:
                    cv2.circle(vis, p, 4, col, -1, lineType=cv2.LINE_AA)
                else:
                    cv2.drawMarker(vis, p, col, markerType=cv2.MARKER_TILTED_CROSS,
                                   markerSize=7, thickness=1, line_type=cv2.LINE_AA)

        # Footer
        foot[:] = 0
        status = f"slot={active_gid+1} ptsA={len(pts_xy_groups[0])} ptsB={len(pts_xy_groups[1])} clouds={[p is not None for p in pcds]} minZ={args.min_depth:.2f} maxZ={args.max_depth:.2f} viz_flip={args.viz_flip}"
        draw_status(foot, status, y=22)

        stack = np.vstack([helpbar, vis, foot])
        cv2.imshow(WIN_NAME, stack)
        k = cv2.waitKey(20) & 0xFF

        if k in (ord('q'), 27):
            break
        elif k == ord('1'):
            active_gid = 0
        elif k == ord('2'):
            active_gid = 1
        elif k == ord('u'):
            if pts_xy_groups[active_gid]:
                pts_xy_groups[active_gid].pop(); labels_groups[active_gid].pop()
        elif k == ord('c'):
            pts_xy_groups[active_gid].clear(); labels_groups[active_gid].clear(); masks[active_gid] = None; pcds[active_gid] = None
        elif k == ord('C'):
            for g in (0,1):
                pts_xy_groups[g].clear(); labels_groups[g].clear(); masks[g] = None; pcds[g] = None

        elif k == ord('r'):
            # run predictor for active slot (one-pass refine for a single mask)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pts_in = np.asarray(pts_xy_groups[active_gid], dtype=np.float32) if pts_xy_groups[active_gid] else None
            labs_in = np.asarray(labels_groups[active_gid], dtype=np.int32)   if labels_groups[active_gid] else None
            box = None
            if rc is not None:
                x0,y0,x1,y1 = rc; box = np.asarray([x0,y0,x1,y1], dtype=np.float32)

            with torch.inference_mode(), (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if (args.bfloat16 and torch.cuda.is_available()) else torch.no_grad()
            ):
                predictor.set_image(rgb)
                masks1, scores1, low1 = predictor.predict(
                    point_coords=pts_in, point_labels=labs_in,
                    box=box, multimask_output=True, return_logits=True,
                    normalize_coords=True,
                )
                best = int(np.argmax(scores1))
                best_low = low1[best:best+1]
                m2, s2, _ = predictor.predict(
                    point_coords=pts_in, point_labels=labs_in,
                    box=box, mask_input=best_low,
                    multimask_output=False, return_logits=False,
                    normalize_coords=True,
                )
            masks[active_gid] = (m2[0] > 0.5).astype(np.uint8)

        elif k == ord('a'):
            # commit current slot’s mask into 3D
            seg = masks[active_gid]
            if seg is None or not seg.any():
                print("No mask in this slot. Press 'r' first."); continue
            col = slot_cols[active_gid]
            rgb_col = (col[2]/255.0, col[1]/255.0, col[0]/255.0)
            p, n = pcd_from_mask(seg.astype(bool), arr,
                                 args.min_depth, args.max_depth,
                                 flip_yz=args.viz_flip,  # only for viewer orientation
                                 color_rgb=rgb_col)
            pcds[active_gid] = p
            print(f"[slot {active_gid+1}] 3D points: {n}")
            pts_xy_groups[active_gid].clear()
            labels_groups[active_gid].clear()

        elif k == ord('v'):
            pcs = [p for p in pcds if p is not None]
            if pcs:
                o3d.visualization.draw_geometries(pcs, window_name="Objects (3D)")
            else:
                print("No 3D clouds yet.")

        elif k == ord('s'):
            # SAMPLE ONE: need both clouds (mask1=trachea, mask2=tumor) committed
            if pcds[0] is None or pcds[1] is None:
                print("Need both clouds committed: slot1=trachea, slot2=tumor. Press 'a' on each first.")
                continue

            tumor_pts = np.asarray(pcds[1].points)
            # Recover camera frame numerically if viz was flipped
            tumor_pts_cam = tumor_pts.copy()
            if args.viz_flip:
                tumor_pts_cam[:,1] *= -1.0
                tumor_pts_cam[:,2] *= -1.0
                
            start_cam, goal_cam, pre_cam = sample_actions_around_tumor_cam(
                tumor_pts_cam,
                x_half=args.x_half,
                y_lo=args.y_lo, y_hi=args.y_hi,
                z_start_lo=args.z_start_lo, z_start_hi=args.z_start_hi,
                z_goal_lo=args.z_goal_lo,   z_goal_hi=args.z_goal_hi,
                z_preshape_lo=args.z_preshape_lo, z_preshape_hi=args.z_preshape_hi
            )

            print("[Random ONE in CAMERA frame]")
            print("  start_cam   (blue): ", start_cam)
            print("  goal_cam    (red)  : ", goal_cam)
            print("  preshape_cam(orange): ", pre_cam)

            # For visualization: mirror if clouds were shown flipped
            one_vis = lambda p: (p if not args.viz_flip else np.array([p[0], -p[1], -p[2]]))
            p_start_vis = one_vis(start_cam)
            p_goal_vis  = one_vis(goal_cam)
            p_pre_vis   = one_vis(pre_cam)

            print("[to VIEWER coords]")
            print("  start_vis   (blue): ", p_start_vis)
            print("  goal_vis    (red)  : ", p_goal_vis)
            print("  preshape_vis(orange): ", p_pre_vis)

            geoms = []
            if pcds[0] is not None: geoms.append(pcds[0])
            if pcds[1] is not None: geoms.append(pcds[1])
            geoms.append(make_sphere_at(p_start_vis, radius=0.001, rgb=(0.1, 0.1, 0.9)))  # blue
            geoms.append(make_sphere_at(p_goal_vis,  radius=0.001, rgb=(0.9, 0.1, 0.1)))  # red
            geoms.append(make_sphere_at(p_pre_vis,   radius=0.001, rgb=(1.0, 0.5, 0.0)))  # orange
            geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02))
            o3d.visualization.draw_geometries(geoms, window_name="Start (G), Goal (R), Preshape (O)")

        elif k == ord('b'):
            # SAMPLE BATCH: need both clouds committed
            if pcds[0] is None or pcds[1] is None:
                print("Need both clouds committed: slot1=trachea, slot2=tumor. Press 'a' on each first.")
                continue

            tumor_pts = np.asarray(pcds[1].points)
            tumor_pts_cam = tumor_pts.copy()
            if args.viz_flip:
                tumor_pts_cam[:,1] *= -1.0
                tumor_pts_cam[:,2] *= -1.0

            start_batch, goal_batch, pre_batch = sample_actions_batch_cam(
                tumor_pts_cam, n=args.n_samples,
                x_half=args.x_half,
                y_lo=args.y_lo, y_hi=args.y_hi,
                z_start_lo=args.z_start_lo, z_start_hi=args.z_start_hi,
                z_goal_lo=args.z_goal_lo,   z_goal_hi=args.z_goal_hi,
                z_preshape_lo=args.z_preshape_lo, z_preshape_hi=args.z_preshape_hi
            )

            print(f"[Random BATCH in CAMERA frame] n={args.n_samples}")
            print("  start_batch   (Nx3) sample[0]:", start_batch[0])
            print("  goal_batch    (Nx3) sample[0]:", goal_batch[0])
            print("  preshape_batch(Nx3) sample[0]:", pre_batch[0])

            # Visualization points (flip if needed)
            start_vis = start_batch if not args.viz_flip else cam_to_view_flip_for_o3d(start_batch.copy())
            goal_vis  = goal_batch  if not args.viz_flip else cam_to_view_flip_for_o3d(goal_batch.copy())
            pre_vis   = pre_batch   if not args.viz_flip else cam_to_view_flip_for_o3d(pre_batch.copy())

            geoms = []
            if pcds[0] is not None: geoms.append(pcds[0])
            if pcds[1] is not None: geoms.append(pcds[1])
            geoms += make_spheres(start_vis, radius=0.0005, rgb=(0.1, 0.1, 0.9))  # blue
            geoms += make_spheres(goal_vis,  radius=0.0005, rgb=(0.9, 0.1, 0.1))  # red
            geoms += make_spheres(pre_vis,   radius=0.0005, rgb=(1.0, 0.5, 0.0))  # orange
            o3d.visualization.draw_geometries(geoms, window_name=f"Batched Samples n={args.n_samples}")

        elif k == ord('g'):
            print("[grab] acquiring new RGB+Cloud …")
            bgr_new, arr_new = grab_pair(args.color, args.cloud, args.slop, args.timeout)
            if bgr_new is None or arr_new is None:
                print("[grab] failed."); continue
            bgr, arr = bgr_new, arr_new
            Hc, Wc = bgr.shape[:2]; Hp, Wp = arr.shape[:2]
            if (Hc, Wc) != (Hp, Wp):
                print(f"[warn] Image {Wc}x{Hc} != cloud {Wp}x{Hp}; using overlap.")

            # reset state
            roi = {"drag": False, "x0": 0, "y0": 0, "x1": 0, "y1": 0}
            for g in (0,1):
                pts_xy_groups[g].clear(); labels_groups[g].clear(); masks[g] = None; pcds[g] = None
            helpbar = np.zeros((HELPBAR_H, Wc, 3), dtype=np.uint8)
            draw_status(helpbar, "Left: FG | Right: BG | Mid drag: ROI | 1/2: slot | r: run | a: add | u: undo | c/C: clear | v: view3D | s: sample | b: batch | g: grab | q: quit", y=22)
            foot = np.zeros((FOOTER_H, Wc, 3), dtype=np.uint8)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
