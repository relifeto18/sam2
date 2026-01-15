#!/usr/bin/env python3
# combined SAM2 segmentation + reprojection into endoscope image
# Usage: python3 sam2_with_reproj.py [--checkpoint ...] [--config ...]
#
# Notes:
# - Edit REGISTRATION_PATH if needed (or pass via env/arg).
# - Expects RealSense:
#     color:  /camera/realsense2_camera_node/color/image_rect_raw
#     cloud:  /camera/realsense2_camera_node/depth/color/points
#     color/camera_info: /camera/realsense2_camera_node/color/camera_info
#   and Endoscope:
#     image_rect: /ves_camera/image_rect
#     camera_info: /ves_camera/camera_info
#
# - Relies on SAM2 code (build_sam / SAM2ImagePredictor) in python path like your original script.

import os, sys, time, threading, argparse
import numpy as np
import cv2
import torch
import open3d as o3d
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo

# ---------- import SAM2 builder/predictor (same as sam2_seg.py) ----------
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ----------------- CONFIG (adjust) -----------------
RS_COLOR_TOPIC = "/camera/realsense2_camera_node/color/image_rect_raw"
RS_CLOUD_TOPIC = "/camera/realsense2_camera_node/depth/color/points"
RS_INFO_TOPIC  = "/camera/realsense2_camera_node/color/camera_info"

ENDO_IMAGE_TOPIC = "/ves_camera/image_rect"
ENDO_INFO_TOPIC  = "/ves_camera/camera_info"

REGISTRATION_PATH = "registration.txt"   # path to T (4x4 csv; RS -> Endoscope)
SLOP_SEC = 0.20
TIMEOUT_SEC = 10.0

# UI params (inspired by your sam2 script)
HELPBAR_H = 32
FOOTER_H = 30
ALPHA = 0.45
WIN_RS = "SAM2 ROI+Clicks (RealSense)"
WIN_ENDO = "Endoscope w/ RS reprojection"

# ---------------------- helpers copied/adapted -----------------------
def stamp_to_float(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9

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
        raise ValueError("Unorganized cloud. Enable ordered cloud in RealSense settings")
    dt = _fields_to_dtype(msg.fields, msg.point_step, msg.is_bigendian)
    data = np.frombuffer(msg.data, dtype=dt, count=msg.width * msg.height)
    if msg.is_bigendian != (sys.byteorder == 'big'):
        data = data.byteswap().newbyteorder()
    return data.reshape((msg.height, msg.width))

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

# ---------------------- Main Node -----------------------
class SAM2ReprojectNode(Node):
    def __init__(self, args):
        super().__init__("sam2_reproj")
        self.bridge = CvBridge()

        # SAM2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"SAM2 device: {self.device}")
        self.model = build_sam2(args.config, args.checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)

        # Registration matrix (RS -> Endoscope)
        if not os.path.exists(args.registration):
            self.get_logger().error(f"Registration file not found: {args.registration}")
            raise RuntimeError("Provide valid registration file.")
        T = np.loadtxt(args.registration, delimiter=",")
        if args.invert_transform:
            self.get_logger().warning("Using inverse of provided registration matrix")
            T = np.linalg.inv(T)
        self.T = T.astype(np.float32)
        self.get_logger().info(f"Loaded transform RS -> Endoscope:\n{self.T}")

        # state (latest)
        self.rs_color = None        # BGR
        self.rs_cloud_arr = None    # structured array (H,W) with x,y,z
        self.rs_info = None         # CameraInfo
        self.endo_image = None      # BGR
        self.endo_info = None       # CameraInfo

        # UI/interaction state (inspired by your original)
        self.roi = {"drag": False, "x0": 0, "y0": 0, "x1": 0, "y1": 0}
        self.pts_xy = []  # list of (x,y) clicks
        self.labels = []  # 1=fg,0=bg
        self.mask = None

        # subscribe to RealSense topics (color + cloud + info)
        self.create_subscription(Image, RS_COLOR_TOPIC, self.rs_color_cb, 10)
        self.create_subscription(PointCloud2, RS_CLOUD_TOPIC, self.rs_cloud_cb, 10)
        self.create_subscription(CameraInfo, RS_INFO_TOPIC, self.rs_info_cb, 10)

        # subscribe to Endoscope topics
        self.create_subscription(Image, ENDO_IMAGE_TOPIC, self.endo_image_cb, 10)
        self.create_subscription(CameraInfo, ENDO_INFO_TOPIC, self.endo_info_cb, 10)

        # UI windows
        cv2.namedWindow(WIN_RS, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_ENDO, cv2.WINDOW_NORMAL)

        # mouse callback for RS window (SAM clicks)
        cv2.setMouseCallback(WIN_RS, self.on_mouse, None)

        # timer to update UI ~30 Hz
        self.create_timer(1.0/30.0, self.ui_timer)

        # helper args
        self.args = args

    # ---- callbacks ----
    def rs_color_cb(self, msg: Image):
        try:
            self.rs_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"cv_bridge (rs color) failed: {e}")

    def rs_cloud_cb(self, msg: PointCloud2):
        try:
            arr = pointcloud2_to_structured_array(msg)
            self.rs_cloud_arr = arr
        except Exception as e:
            self.get_logger().warn(f"PointCloud2 parse failed: {e}")

    def rs_info_cb(self, msg: CameraInfo):
        self.rs_info = msg

    def endo_image_cb(self, msg: Image):
        try:
            self.endo_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"cv_bridge (endo image) failed: {e}")

    def endo_info_cb(self, msg: CameraInfo):
        self.endo_info = msg

    # ---- mouse for SAM clicks / ROI ----
    def on_mouse(self, event, x, y, flags, param):
        # only accept clicks inside image (no helpbar here)
        if self.rs_color is None: return
        Hc, Wc = self.rs_color.shape[:2]
        if not (0 <= x < Wc and 0 <= y < Hc):
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pts_xy.append((x, y)); self.labels.append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.pts_xy.append((x, y)); self.labels.append(0)
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.roi.update({"drag": True, "x0": x, "y0": y, "x1": x, "y1": y})
        elif event == cv2.EVENT_MOUSEMOVE and self.roi["drag"]:
            self.roi["x1"], self.roi["y1"] = x, y
        elif event == cv2.EVENT_MBUTTONUP:
            self.roi["drag"] = False; self.roi["x1"], self.roi["y1"] = x, y

    # ---- mask → 3D lifting (per-pixel) ----
    def mask_pixels_to_rs_points(self, mask_bool):
        """
        Given a boolean mask aligned to rs_color (H,W), return Nx3 RS-frame points and their pixel coords.
        Uses self.rs_cloud_arr's x,y,z fields.
        """
        if self.rs_cloud_arr is None:
            return np.zeros((0,3)), np.zeros((0,2), dtype=int)
        Hc, Wc = mask_bool.shape
        Hp, Wp = self.rs_cloud_arr.shape[:2]
        # use overlap rectangle
        xo1, yo1 = min(Wc, Wp), min(Hc, Hp)
        seg = mask_bool[:yo1, :xo1]
        # build xyz array
        xyz = np.stack([self.rs_cloud_arr['x'][:yo1, :xo1],
                        self.rs_cloud_arr['y'][:yo1, :xo1],
                        self.rs_cloud_arr['z'][:yo1, :xo1]], axis=-1)
        z = xyz[..., 2]
        finite = np.isfinite(xyz).all(axis=-1)
        valid = seg & finite & (z > self.args.min_depth) & (z < self.args.max_depth)
        if not valid.any():
            return np.zeros((0,3)), np.zeros((0,2), dtype=int)
        pts = xyz[valid].reshape(-1,3).astype(np.float32)
        ys, xs = np.where(valid)
        pix = np.stack([xs, ys], axis=1).astype(int)
        return pts, pix

    # ---- transform and project into endoscope image ----
    def reproject_mask_to_endoscope(self, mask_bool):
        """Return an overlayed endoscope BGR image (copy) with RS mask reprojected into it."""
        if self.endo_image is None or self.rs_cloud_arr is None or self.endo_info is None or self.rs_info is None:
            return None

        overlay = self.endo_image.copy()
        # get RS->ENDO transform
        T = self.T  # 4x4
        # endoscope projection matrix P (3x4) from CameraInfo P
        P_e = np.array(self.endo_info.p, dtype=np.float32).reshape(3,4)
        K_e = P_e[:, :3]
        fx_e, fy_e = K_e[0,0], K_e[1,1]
        cx_e, cy_e = K_e[0,2], K_e[1,2]

        # RS intrinsics (for reference/back-projection) from rs_info
        K_rs = np.array(self.rs_info.k, dtype=np.float32).reshape(3,3)
        fx_r, fy_r = K_rs[0,0], K_rs[1,1]
        cx_r, cy_r = K_rs[0,2], K_rs[1,2]

        # convert mask → RS points
        pts_rs, pix = self.mask_pixels_to_rs_points(mask_bool)
        if pts_rs.shape[0] == 0:
            return overlay

        # append homogeneous coordinate and transform
        ones = np.ones((pts_rs.shape[0],1), dtype=np.float32)
        pts_rs_h = np.concatenate([pts_rs, ones], axis=1)   # (N,4)
        pts_e_h = (T @ pts_rs_h.T).T                        # (N,4)
        Xe, Ye, Ze = pts_e_h[:,0], pts_e_h[:,1], pts_e_h[:,2]

        # project and color overlay
        H_e, W_e = overlay.shape[:2]
        # choose a color for overlay
        color = (0, 200, 255)  # BGR
        for i in range(pts_rs.shape[0]):
            ze = Ze[i]
            if ze <= 0 or not np.isfinite(ze):
                continue
            ue = int(round(fx_e * Xe[i] / ze + cx_e))
            ve = int(round(fy_e * Ye[i] / ze + cy_e))
            if 0 <= ue < W_e and 0 <= ve < H_e:
                # blend pixel (simple alpha)
                overlay[ve, ue] = ((1-ALPHA)*overlay[ve, ue] + ALPHA*np.array(color, dtype=np.uint8)).astype(np.uint8)

        return overlay

    # ---- UI tick ----
    def ui_timer(self):
        # must have an RS color to show; otherwise nothing to do
        if self.rs_color is None:
            # still try to show endo overlay if available (no mask)
            if self.endo_image is not None:
                cv2.imshow(WIN_ENDO, self.endo_image)
                cv2.waitKey(1)
            return

        bgr = self.rs_color.copy()

        # draw current mask overlay on RS
        vis = overlay_mask(bgr, self.mask, color_bgr=(40,220,40), alpha=0.30)

        # draw ROI rectangle if any
        rc = None
        if self.roi and self.roi["x1"] != self.roi["x0"] and self.roi["y1"] != self.roi["y0"]:
            x0,y0,x1,y1 = self.roi["x0"], self.roi["y0"], self.roi["x1"], self.roi["y1"]
            cv2.rectangle(vis, (x0,y0), (x1,y1), (0,255,255), 2)

        # draw clicks
        for (x,y), lab in zip(self.pts_xy, self.labels):
            if lab == 1:
                cv2.circle(vis, (int(x),int(y)), 4, (40,220,40), -1, lineType=cv2.LINE_AA)
            else:
                cv2.drawMarker(vis, (int(x),int(y)), (0,0,255), markerType=cv2.MARKER_TILTED_CROSS,
                               markerSize=7, thickness=1, line_type=cv2.LINE_AA)

        # footer text
        Hc, Wc = vis.shape[:2]
        helpbar = np.zeros((HELPBAR_H, Wc, 3), dtype=np.uint8)
        foot = np.zeros((FOOTER_H, Wc, 3), dtype=np.uint8)
        cv2.putText(helpbar, "Left: FG | Right: BG | Mid: ROI | r: run SAM | a: commit mask | g: clear mask | q: quit",
                    (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1, cv2.LINE_AA)
        status = f"mask_pts={len(self.pts_xy)} rs_ready={self.rs_cloud_arr is not None} endo_ready={self.endo_image is not None}"
        cv2.putText(foot, status, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1, cv2.LINE_AA)

        stack = np.vstack([helpbar, vis, foot])
        cv2.imshow(WIN_RS, stack)

        # Show endoscope overlay with reprojected mask (if any mask present)
        if self.mask is not None:
            endo_overlay = self.reproject_mask_to_endoscope(self.mask.astype(bool))
            if endo_overlay is not None:
                cv2.imshow(WIN_ENDO, endo_overlay)
        else:
            if self.endo_image is not None:
                cv2.imshow(WIN_ENDO, self.endo_image)

        # handle keypresses (same keys as before)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:
            # exit gracefully
            cv2.destroyAllWindows()
            rclpy.shutdown()
            sys.exit(0)
        elif k == ord('r'):
            # run SAM on current RS image and current points/ROI
            if self.rs_color is None:
                self.get_logger().warning("No RS image for SAM")
                return
            rgb = cv2.cvtColor(self.rs_color, cv2.COLOR_BGR2RGB)
            pts_in = np.asarray(self.pts_xy, dtype=np.float32) if self.pts_xy else None
            labs_in = np.asarray(self.labels, dtype=np.int32) if self.labels else None
            box = None
            if self.roi and (self.roi["x1"] != self.roi["x0"] and self.roi["y1"] != self.roi["y0"]):
                x0,y0,x1,y1 = self.roi["x0"], self.roi["y0"], self.roi["x1"], self.roi["y1"]
                box = np.asarray([x0,y0,x1,y1], dtype=np.float32)
            with torch.inference_mode(), (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if (self.args.bfloat16 and torch.cuda.is_available()) else torch.no_grad()
            ):
                self.predictor.set_image(rgb)
                masks1, scores1, low1 = self.predictor.predict(
                    point_coords=pts_in, point_labels=labs_in,
                    box=box, multimask_output=True, return_logits=True,
                    normalize_coords=True,
                )
                if masks1 is None or len(masks1) == 0:
                    self.get_logger().warning("SAM returned no masks")
                    return
                best = int(np.argmax(scores1))
                best_low = low1[best:best+1]
                m2, s2, _ = self.predictor.predict(
                    point_coords=pts_in, point_labels=labs_in,
                    box=box, mask_input=best_low,
                    multimask_output=False, return_logits=False,
                    normalize_coords=True,
                )
                self.mask = (m2[0] > 0.5).astype(np.uint8)
                self.get_logger().info("SAM produced mask (press 'a' to commit/show in 3D or overlay reproj)")
        elif k == ord('a'):
            # commit mask => optionally show 3D pointcloud (visualization)
            if self.mask is None:
                self.get_logger().warning("No mask to commit")
            else:
                # lift to 3D and show in Open3D windows (just like sam2_seg 'a' key)
                seg_bool = self.mask.astype(bool)
                pcd, n = self.pcd_from_mask_for_view(seg_bool)
                if pcd is not None:
                    o3d.visualization.draw_geometries([pcd], window_name="Committed mask (3D)")
                    self.get_logger().info(f"Committed mask -> {n} points (viewer)")
                else:
                    self.get_logger().warning("No valid 3D points for this mask")
        elif k == ord('g'):
            # clear current mask & clicks
            self.pts_xy.clear(); self.labels.clear(); self.mask = None

    def pcd_from_mask_for_view(self, seg_full_bool, flip_yz=True, color_rgb=(1.0, 0.5, 0.0)):
        """Re-uses pcd_from_mask logic (visualization flip optional)."""
        if self.rs_cloud_arr is None:
            return None, 0
        Hp, Wp = self.rs_cloud_arr.shape[:2]
        Hc, Wc = seg_full_bool.shape
        xo1, yo1 = min(Wc, Wp), min(Hc, Hp)
        seg_crop = seg_full_bool[:yo1, :xo1]
        xyz_roi = np.stack([self.rs_cloud_arr['x'][:yo1, :xo1],
                            self.rs_cloud_arr['y'][:yo1, :xo1],
                            self.rs_cloud_arr['z'][:yo1, :xo1]], axis=-1)
        z = xyz_roi[..., 2]
        finite = np.isfinite(xyz_roi).all(axis=-1)
        depth_ok = (z > self.args.min_depth) & (z < self.args.max_depth)
        valid = seg_crop & finite & depth_ok
        if not valid.any(): return None, 0
        pts = xyz_roi[valid].reshape(-1,3).astype(np.float64)
        if flip_yz:
            pts[:,1] *= -1.0
            pts[:,2] *= -1.0
        cols = np.tile(np.array(color_rgb, dtype=np.float64)[None,:], (pts.shape[0],1))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        return pcd, pts.shape[0]

def main():
    ap = argparse.ArgumentParser("SAM2 + Reproject to Endoscope")
    ap.add_argument('--checkpoint', type=str, default="/home/desser/sam2/checkpoints/sam2.1_hiera_large.pt")
    ap.add_argument('--config', type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument('--registration', type=str, default="/home/desser/aliss_core/src/realsense/eye2eye/registration.txt")
    ap.add_argument('--invert_transform', action='store_true')
    ap.add_argument('--bfloat16', action='store_true')
    ap.add_argument('--min_depth', type=float, default=0.05)
    ap.add_argument('--max_depth', type=float, default=2.0)
    args = ap.parse_args()

    rclpy.init()
    node = SAM2ReprojectNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
