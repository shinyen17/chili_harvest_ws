"""Chili perception node.

Pipeline per colour frame:
  1. Ripeness detector (YOLO detect)      -> boxes labelled ripe / semi-ripe / unripe
  2. Chili + stem segmentation (YOLO seg) -> chili masks, stem masks (full frame)
  3. Each ripeness box -> matching chili mask (IoU) -> attached stem mask
  4. Cut pixel on the stem -> depth (stem, else chili body) -> 3D point
  5. TF: camera optical frame -> base_link

Publishes
  /chili/harvest_target  geometry_msgs/PointStamped  (base frame, ripe chilies only, nearest one)
  /chili/detections      std_msgs/String             (JSON list: label, conf, box, cut point ...)
  /chili/stats           std_msgs/String             (JSON counts + inference time)
  /chili/debug_image     sensor_msgs/Image           (annotated frame)
"""
import json
import os
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_geometry_msgs import do_transform_point  # noqa: F401  (registers PointStamped)
from tf2_ros import Buffer, TransformListener
from ultralytics import YOLO

from chili_harvest import vision_utils as vu

COLORS = {  # BGR
    'ripe': (40, 40, 230),
    'semiripe': (0, 165, 255),
    'unripe': (60, 200, 60),
}


class ChiliPerception(Node):
    def __init__(self):
        super().__init__('chili_perception')

        p = self.declare_parameter
        p('ripeness_model', 'chili_ripeness.pt')
        p('seg_model', 'chili_stem_seg.pt')
        p('ripe_classes', ['ripe'])
        p('chili_class', 'chili')
        p('stem_class', 'stem')
        p('ripeness_conf', 0.5)
        p('seg_conf', 0.25)
        p('imgsz', 640)
        p('device', '')                    # '' = auto, 'cpu', '0' = first GPU
        p('color_topic', '/camera/color/image_raw')
        p('depth_topic', '/camera/depth/image_raw')
        p('info_topic', '/camera/color/camera_info')
        p('base_frame', 'base_link')
        p('camera_frame', '')              # '' = take frame_id from camera_info
        p('min_depth_mm', 150)
        p('max_depth_mm', 1500)
        p('max_depth_jump_mm', 30.0)
        p('flip_depth_horizontal', False)  # cocoa code flipped Astra depth; check!
        p('stem_cut_ratio', 0.5)           # 0 = at the chili, 1 = far end of stem
        p('stem_gap_px', 12)
        p('show_window', True)
        p('show_depth_overlay', False)
        p('process_every_n', 1)

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.gp = gp

        self.model_ripe = YOLO(self._model_path(gp('ripeness_model')))
        self.model_seg = YOLO(self._model_path(gp('seg_model')))
        self.ripe_set = {vu.normalize_label(c) for c in gp('ripe_classes')}
        self.chili_cls = vu.normalize_label(gp('chili_class'))
        self.stem_cls = vu.normalize_label(gp('stem_class'))
        self.get_logger().info(f'Ripeness classes: {self.model_ripe.names}')
        self.get_logger().info(f'Segmentation classes: {self.model_seg.names}')
        self._check_class_names()

        self.bridge = CvBridge()
        self.K = None
        self.cam_frame = gp('camera_frame') or None
        self.depth = None
        self.depth_warned = False
        self.frame_i = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, gp('color_topic'), self.on_color, sensor_qos)
        self.create_subscription(Image, gp('depth_topic'), self.on_depth, sensor_qos)
        self.create_subscription(CameraInfo, gp('info_topic'), self.on_info, 10)

        self.pub_target = self.create_publisher(PointStamped, '/chili/harvest_target', 10)
        self.pub_dets = self.create_publisher(String, '/chili/detections', 10)
        self.pub_stats = self.create_publisher(String, '/chili/stats', 10)
        self.pub_img = self.create_publisher(Image, '/chili/debug_image', 1)

        if gp('show_window'):
            cv2.namedWindow('Chili perception', cv2.WINDOW_NORMAL)
        self.get_logger().info('Chili perception ready, waiting for camera...')

    # ---------------------------------------------------------------- setup
    def _model_path(self, name):
        if os.path.isabs(name) and os.path.exists(name):
            return name
        path = os.path.join(get_package_share_directory('chili_harvest'), 'models', name)
        if not os.path.exists(path):
            raise FileNotFoundError(f'Model not found: {path}. Copy your .pt into '
                                    'src/chili_harvest/models/ and rebuild.')
        return path

    def _check_class_names(self):
        ripe_names = {vu.normalize_label(n) for n in self.model_ripe.names.values()}
        seg_names = {vu.normalize_label(n) for n in self.model_seg.names.values()}
        if not self.ripe_set & ripe_names:
            self.get_logger().error(f'None of ripe_classes {self.ripe_set} is in the ripeness '
                                    f'model {ripe_names}. Fix the parameter.')
        for c in (self.chili_cls, self.stem_cls):
            if c not in seg_names:
                self.get_logger().error(f"Class '{c}' not in segmentation model {seg_names}.")

    # ------------------------------------------------------------ callbacks
    def on_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            if self.cam_frame is None:
                self.cam_frame = msg.header.frame_id
            self.get_logger().info(f'Camera intrinsics received, frame = {self.cam_frame}')

    def on_depth(self, msg):
        d = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if d.dtype.kind == 'f':          # 32FC1 in metres -> millimetres
            d = np.nan_to_num(d * 1000.0).astype(np.uint16)
        if self.gp('flip_depth_horizontal'):
            d = cv2.flip(d, 1)
        self.depth = d

    def on_color(self, msg):
        self.frame_i += 1
        if self.K is None or self.depth is None:
            return
        if self.frame_i % max(1, self.gp('process_every_n')):
            return
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w = img.shape[:2]
        depth = self.depth
        if depth.shape[:2] != (h, w):
            if not self.depth_warned:
                self.get_logger().warn(
                    f'Depth {depth.shape[:2]} != colour {(h, w)}; resizing. Turn on depth '
                    'registration in the camera driver so pixels line up.')
                self.depth_warned = True
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        t0 = time.time()
        kw = dict(imgsz=self.gp('imgsz'), verbose=False)
        if self.gp('device'):
            kw['device'] = self.gp('device')
        r_ripe = self.model_ripe(img, conf=self.gp('ripeness_conf'), **kw)[0]
        r_seg = self.model_seg(img, conf=self.gp('seg_conf'), retina_masks=True, **kw)[0]
        infer_ms = (time.time() - t0) * 1000.0

        chili_boxes, chili_masks, stem_masks = self._split_seg(r_seg, h, w)
        dets, counts = [], {'ripe': 0, 'semiripe': 0, 'unripe': 0, 'other': 0}
        vis = img.copy()
        if self.gp('show_depth_overlay'):
            dv = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255.0 / 1500), cv2.COLORMAP_JET)
            vis = cv2.addWeighted(vis, 0.5, dv, 0.5, 0)

        for box, cls, conf in zip(r_ripe.boxes.xyxy.cpu().numpy(),
                                  r_ripe.boxes.cls.cpu().numpy().astype(int),
                                  r_ripe.boxes.conf.cpu().numpy()):
            label = vu.normalize_label(self.model_ripe.names[cls])
            counts[label if label in counts else 'other'] += 1
            ripe = label in self.ripe_set
            det = {'label': self.model_ripe.names[cls], 'conf': round(float(conf), 3),
                   'box': [int(v) for v in box], 'ripe': ripe}
            self._draw_box(vis, box, label, conf)
            if ripe:
                det.update(self._locate_cut(img, depth, box, chili_boxes, chili_masks,
                                            stem_masks, vis))
            dets.append(det)

        target = self._choose_target(dets)
        if target is not None:
            pt = PointStamped()
            pt.header.stamp = msg.header.stamp
            pt.header.frame_id = self.gp('base_frame')
            pt.point.x, pt.point.y, pt.point.z = target['cut_base']
            self.pub_target.publish(pt)

        stats = dict(counts, stems_matched=sum('cut_px' in d for d in dets),
                     targets=int(target is not None), infer_ms=round(infer_ms, 1))
        self.pub_dets.publish(String(data=json.dumps(dets)))
        self.pub_stats.publish(String(data=json.dumps(stats)))
        cv2.putText(vis, f"ripe {counts['ripe']}  semi {counts['semiripe']}  unripe "
                    f"{counts['unripe']}  {infer_ms:.0f} ms", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        self.pub_img.publish(self.bridge.cv2_to_imgmsg(vis, 'bgr8'))
        if self.gp('show_window'):
            cv2.imshow('Chili perception', vis)
            cv2.waitKey(1)

    # -------------------------------------------------------------- helpers
    def _split_seg(self, r, h, w):
        chili_boxes, chili_masks, stem_masks = [], [], []
        if r.masks is None:
            return chili_boxes, chili_masks, stem_masks
        masks = r.masks.data.cpu().numpy()
        for i, cls in enumerate(r.boxes.cls.cpu().numpy().astype(int)):
            m = masks[i]
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            m = m > 0.5
            name = vu.normalize_label(self.model_seg.names[cls])
            if name == self.chili_cls:
                chili_boxes.append(r.boxes.xyxy[i].cpu().numpy())
                chili_masks.append(m)
            elif name == self.stem_cls:
                stem_masks.append(m)
        return chili_boxes, chili_masks, stem_masks

    def _locate_cut(self, img, depth, box, chili_boxes, chili_masks, stem_masks, vis):
        out = {}
        ci = vu.match_chili(box, chili_boxes)
        chili_mask = chili_masks[ci] if ci is not None else None
        si, attach = vu.select_stem(stem_masks, chili_mask=chili_mask, chili_box=box,
                                    gap_px=self.gp('stem_gap_px'))
        if si is None:
            out['reason'] = 'no stem attached'
            return out
        stem = stem_masks[si]
        vis[stem] = (0.5 * vis[stem] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        px = vu.cut_pixel(stem, attach, self.gp('stem_cut_ratio'))
        if px is None:
            out['reason'] = 'empty stem mask'
            return out
        u, v = px
        out['cut_px'] = [u, v]

        lo, hi = self.gp('min_depth_mm'), self.gp('max_depth_mm')
        kernel = np.ones((5, 5), np.uint8)
        stem_area = cv2.dilate(stem.astype(np.uint8), kernel).astype(bool)
        z_stem = vu.robust_depth_mm(depth, stem_area, lo, hi)
        if chili_mask is not None:
            body = chili_mask
        else:
            body = vu._box_mask(depth.shape[:2], box)
        z_chili = vu.robust_depth_mm(depth, body, lo, hi)
        z_mm, src = vu.fuse_depth(z_stem, z_chili, self.gp('max_depth_jump_mm'))
        if z_mm is None:
            out['reason'] = 'no valid depth'
            cv2.circle(vis, (u, v), 6, (0, 0, 255), 2)
            return out
        z_m = z_mm / 1000.0
        cam = vu.deproject(u, v, z_m, self.K)
        out['cut_cam'] = [round(c, 4) for c in cam]
        out['depth_source'] = src

        base = self._to_base(cam)
        if base is None:
            out['reason'] = 'no TF to base'
        else:
            out['cut_base'] = [round(c, 4) for c in base]
        cv2.circle(vis, (u, v), 6, (0, 255, 255), -1)
        if base is not None:
            cv2.putText(vis, f'cut ({base[0]:.2f},{base[1]:.2f},{base[2]:.2f}) m',
                        (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        return out

    def _to_base(self, cam_xyz):
        pt = PointStamped()
        pt.header.frame_id = self.cam_frame
        pt.point.x, pt.point.y, pt.point.z = cam_xyz
        try:
            tf = self.tf_buffer.lookup_transform(self.gp('base_frame'), self.cam_frame,
                                                 rclpy.time.Time())
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'TF {self.cam_frame} -> base: {e}',
                                   throttle_duration_sec=5.0)
            return None
        p = do_transform_point(pt, tf).point
        return (p.x, p.y, p.z)

    def _choose_target(self, dets):
        # nearest ripe chili with a valid cut point; reach is checked by the controller
        best, best_d = None, float('inf')
        for d in dets:
            if d.get('ripe') and 'cut_base' in d:
                dist = float(np.linalg.norm(d['cut_base']))
                if dist < best_d:
                    best, best_d = d, dist
        return best

    def _draw_box(self, vis, box, label, conf):
        x1, y1, x2, y2 = [int(v) for v in box]
        c = COLORS.get(label, (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
        cv2.putText(vis, f'{label} {conf:.2f}', (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)


def main(args=None):
    rclpy.init(args=args)
    node = ChiliPerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
