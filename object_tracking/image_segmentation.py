import torch
import os
import numpy as np
from ament_index_python.packages import get_package_share_directory
import cv2
from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference import load_model, predict

import requests
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from PIL import Image as PILImage

import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Point, PoseStamped

import math
from geometry_msgs.msg import Quaternion

import time

class SAMSegmentor:
    def __init__(self, hfov=70, vfov=40):
        self.HFOV = hfov
        self.VFOV = vfov

        share_dir = get_package_share_directory('object_tracking')
        checkpoint_path_SAM = os.path.join(share_dir, 'model_weights', 'sam_vit_h_4b8939.pth')

        if not os.path.isfile(checkpoint_path_SAM):
            raise FileNotFoundError(f"\n[ERROR] SAM checkpoint not found at:\n  {checkpoint_path_SAM}\n\n"
                                    f"Please download it from:\n"
                                    f"  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth\n"
                                    f"and place it in:\n  object_tracking/models/")

        # SAM + DINO
        self.sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path_SAM).to("cuda")
        self.predictor = SamPredictor(self.sam)
        self.dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to("cuda")

    def segment(self, image_bgr, prompt, depth_map):
        print("segmenting...")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = PILImage.fromarray(image_rgb)
        text_labels = [[prompt]]

        start_time_DINO = time.time()

        inputs = self.dino_processor(images=image_pil, text=text_labels, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        print("received outputs from DINO")

        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image_pil.size[::-1]]
        )

        print("dino_processor finished")

        end_time_DINO = time.time()

        DINO_time = end_time_DINO - start_time_DINO

        print(f"GroundingDINO's work time is {DINO_time}")

        result = results[0]

        # Фильтрация по порогу
        box_threshold = 0.75
        filtered = [
            (box.cpu().numpy(), score.item(), label)
            for box, score, label in zip(result["boxes"], result["scores"], result["labels"])
            if score.item() >= box_threshold
        ]

        if not filtered:
            print("Объект не найден по уверенности")
            return image_bgr, None, depth_map, 0

        # Выбери самый уверенный бокс
        box, score, label = sorted(filtered, key=lambda x: -x[1])[0]
        input_box = np.array([box])
        print(f"Найден объект: {label} (score={score:.2f})")

        print("Received bounding boxes")

        start_time_SAM = time.time()

        input_boxes = self.predictor.transform.apply_boxes_torch(torch.tensor(input_box), image_bgr.shape[:2]).numpy()
        self.predictor.set_image(image_rgb)
        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=torch.tensor(input_boxes).to("cuda"),
            multimask_output=False,
        )

        end_time_SAM = time.time()

        SAM_time = end_time_SAM - start_time_SAM

        print(f"SAM's work time is {SAM_time}")

        print("masks acquired")

        mask = masks[0][0].cpu().numpy()
        ys, xs = np.where(mask)
        if xs.size == 0 or ys.size == 0:
            return image_bgr, None, depth_map

        center_coords = self.get_center_coordinates(mask)

        image_out = image_bgr.copy()
        image_out[mask > 0] = (0, 255, 0)

        print("masked image received, returning")

        for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
            if score.item() >= box_threshold:
                x1, y1, x2, y2 = box.cpu().numpy()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(image_out, (x1, y1), (x2, y2), (255, 0, 0), 2)
                text = f"{label} ({score:.2f})"
                cv2.putText(image_out, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return image_out, center_coords, depth_map, DINO_time+SAM_time
    
    def get_center_coordinates(self, mask):
        y_indices, x_indices = np.where(mask)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None
        x_mean = int(np.mean(x_indices))
        y_mean = int(np.mean(y_indices))
        return (x_mean, y_mean)
    
    def get_goal_point(self, depth_image, center_coords, camera_transform, transform_base, camera_intrinsics, offset):
        fx, fy, cx, cy = camera_intrinsics
        
        x_px = int(center_coords[0])
        y_px = int(center_coords[1])
        
        depth = depth_image[y_px, x_px]

        X = (x_px - cx) * depth / fx
        Y = (y_px - cy) * depth / fy
        Z = depth

        point_camera = Point()
        point_camera.x = X
        point_camera.y = Y
        point_camera.z = float(Z)

        point_stamped = tf2_geometry_msgs.PointStamped()
        point_stamped.header.frame_id = 'depth_camera_link_optical'
        point_stamped.header.stamp = camera_transform.header.stamp
        point_stamped.point = point_camera

        point_world = tf2_geometry_msgs.do_transform_point(point_stamped, camera_transform)

        robot_x = transform_base.transform.translation.x
        robot_y = transform_base.transform.translation.y

        dx = point_world.point.x - robot_x
        dy = point_world.point.y - robot_y

        distance = np.hypot(dx, dy)

        if distance <= offset:
            goal_x = robot_x
            goal_y = robot_y

            goal = PoseStamped()
                    
            goal.header.frame_id = 'map'

            goal.pose.position.x = goal_x
            goal.pose.position.y = goal_y
            return goal

        #scale = (distance - offset) / distance
        scale = 0.8

        goal_x = robot_x + dx * scale
        goal_y = robot_y + dy * scale

        print(f'Объект в map frame: X={point_world.point.x:.2f}, Y={point_world.point.y:.2f}, Z={point_world.point.z:.2f}')
        print(f'Расстояние до цели distance = {distance:.2f}, offset = {offset:.2f}')

        goal = PoseStamped()
                    
        goal.header.frame_id = 'map'

        theta = np.arctan2(dy, dx)

        def yaw_to_quaternion(yaw):
            q = Quaternion()
            q.z = math.sin(yaw / 2.0)
            q.w = math.cos(yaw / 2.0)
            return q

        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y

        goal.pose.orientation = yaw_to_quaternion(theta)

        return goal