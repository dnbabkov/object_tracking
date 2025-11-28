import math
import time

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, PoseStamped, Twist, Quaternion, PointStamped

from cv_bridge import CvBridge, CvBridgeError

from object_tracking.image_segmentation import SAMSegmentor
from object_tracking.clip_image_segmentation import CLIPSegmentor

from message_filters import ApproximateTimeSynchronizer, Subscriber

import tf2_ros
from tf2_geometry_msgs import do_transform_point

from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus


class SAMNode(Node):
    def __init__(self):
        super().__init__('tracker_node')

        self.get_logger().info('Node initialized')

        # Состояние
        self.camera_info_received = False
        self.target_reached = False
        self.target_found = False
        self.SAM = True  # переключение между SAM и CLIP

        # Объекты
        self.bridge = CvBridge()
        if self.SAM:
            self.segmentor = SAMSegmentor()
            self.goal_position = None
        else:
            self.segmentor = CLIPSegmentor()

        # Текущее состояние/параметры
        self.current_prompt = None
        self.current_pose = None
        self.offset = 0.5
        self.offset_delta = 0.35
        self.total_seg_time = 0.0
        self.segmentations = 0
        self.latest_depth = None

        # Параметры поиска и обновления цели
        self.declare_parameter('search_angular_speed', 0.5)
        self.search_angular_speed = (
            self.get_parameter('search_angular_speed')
            .get_parameter_value()
            .double_value
        )

        self.declare_parameter('goal_update_period', 2.0)  # сек, для CLIP
        self.goal_update_period = (
            self.get_parameter('goal_update_period')
            .get_parameter_value()
            .double_value
        )

        # Время последнего обновления цели для CLIP
        self.last_goal_update_time = self.get_clock().now() - Duration(
            seconds=self.goal_update_period * 2.0
        )

        # Подписчики (синхронные RGB + depth)
        self.rgb_sub = Subscriber(
            self,
            Image,
            '/image_in',
            qos_profile=rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value,
        )
        self.depth_sub = Subscriber(
            self,
            Image,
            '/depth_camera/depth/image_raw',
            qos_profile=rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value,
        )

        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1,
        )
        self.ts.registerCallback(self.synced_image_depth_callback)

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            1,
        )

        self.prompt_sub = self.create_subscription(
            String,
            '/target_prompt',
            self.prompt_callback,
            1,
        )

        # Публикаторы
        self.search_cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_pub = self.create_publisher(Image, '/image_out', 1)
        self.pose_pub = self.create_publisher(PoseStamped, '/goal_pose', 1)

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Nav2
        self.nav2_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.nav2_goal_handle = None
        self.nav2_goal_active = False

        self.pending_goal_pose = None

        # Таймер поиска
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    # -------------------------------------------------------------------------
    # Вспомогательные методы
    # -------------------------------------------------------------------------
    @staticmethod
    def yaw_to_quaternion(yaw: float) -> Quaternion:
        """Преобразование yaw в кватернион (2D поворот вокруг Z)."""
        q = Quaternion()
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

    def _lookup_transform(self, target_frame: str, source_frame: str, timeout_sec: float = 0.5):
        """Обёртка над lookup_transform, чтобы не дублировать код."""
        return self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=timeout_sec),
        )

    def _publish_search_rotation(self):
        """Публикация команды вращения в режиме поиска цели."""
        msg = Twist()
        msg.angular.z = self.search_angular_speed
        self.search_cmd_pub.publish(msg)

    def _add_segmentation_time(self, segmentation_time: float):
        """Накопление статистики по времени сегментации."""
        self.total_seg_time += segmentation_time
        self.segmentations += 1

    def _log_avg_seg_time(self, prefix: str):
        """Логирование и сброс средней продолжительности сегментации."""
        if self.segmentations == 0:
            return
        avg_time = self.total_seg_time / self.segmentations
        self.get_logger().info(f'{prefix} {avg_time}')
        self.total_seg_time = 0.0
        self.segmentations = 0

    def _pixel_to_camera_point(self, x_px: int, y_px: int, depth_image) -> Point | None:
        """
        Проекция пикселя (x_px, y_px) + depth в 3D-точку в системе координат камеры.
        Возвращает None, если глубина некорректна.
        """
        depth_value = depth_image[y_px, x_px]

        if np.isnan(depth_value) or depth_value <= 0.0:
            return None

        X = (x_px - self.cx) * depth_value / self.fx
        Y = (y_px - self.cy) * depth_value / self.fy
        Z = depth_value

        point_camera = Point()
        point_camera.x = X
        point_camera.y = Y
        point_camera.z = float(Z)

        return point_camera

    def _camera_point_to_world(
        self,
        point_camera: Point,
        camera_frame: str = 'depth_camera_link_optical',
        world_frame: str = 'map',
        timeout_sec: float = 0.5,
    ) -> Point:
        """
        Трансформация точки из координат камеры в мировые координаты (world_frame).
        """
        point_stamped = PointStamped()
        point_stamped.header.frame_id = camera_frame
        point_stamped.header.stamp = self.get_clock().now().to_msg()
        point_stamped.point = point_camera

        transform = self._lookup_transform(
            world_frame,
            camera_frame,
            timeout_sec=timeout_sec,
        )

        point_world_stamped = do_transform_point(point_stamped, transform)
        return point_world_stamped.point


    # --- Nav2 ---------------------------------------------------------------
    def _send_goal_to_nav2(self, goal_pose: PoseStamped):
        """Отправка цели в Nav2 + публикация в топик."""
        # Для совместимости/визуализации оставляем публикацию в топик
        self.pose_pub.publish(goal_pose)

        if not self.nav2_client.wait_for_server(timeout_sec=0.5):
            self.get_logger().warn('Nav2 action server not available')
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        send_goal_future = self.nav2_client.send_goal_async(
            goal_msg,
            feedback_callback=self._nav2_feedback_callback,
        )
        send_goal_future.add_done_callback(self._nav2_goal_response_callback)

        self.nav2_goal_active = True
        self.target_reached = False

    def _nav2_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        # Можно логировать расстояние для отладки
        # self.get_logger().debug(
        #     f'Nav2 feedback: distance_remaining = {feedback.distance_remaining:.2f}'
        # )
        pass

    def _nav2_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Nav2 goal rejected')
            self.nav2_goal_active = False
            self.nav2_goal_handle = None
            return

        self.get_logger().info('Nav2 goal accepted')
        self.nav2_goal_handle = goal_handle

        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self._nav2_result_callback)


    def _update_nav2_goal(self, new_goal_pose: PoseStamped):
        """
        Обновить Nav2-цель:
        - если активной цели нет — просто отправляем новую;
        - если есть — запоминаем новую и просим Nav2 отменить старую.
          После прихода результата с STATUS_CANCELED отправляем новую.
        """
        # Если цели нет или handle потерян — просто шлём новую
        if not self.nav2_goal_active or self.nav2_goal_handle is None:
            self.pending_goal_pose = None
            self._send_goal_to_nav2(new_goal_pose)
            return

        # Цель активна — запросим отмену, а новую отправим после CANCEL результата
        self.pending_goal_pose = new_goal_pose
        self.get_logger().info('Requesting Nav2 goal cancel for update...')
        cancel_future = self.nav2_goal_handle.cancel_goal_async()
        # отдельный callback не нужен: результат придёт в _nav2_result_callback
    
    def _nav2_result_callback(self, future):
        result = future.result()
        status = result.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Nav2: goal reached')
            self.target_reached = True
            self._log_avg_seg_time('Average segmentation time is')

        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info('Nav2: goal canceled')

            # Если у нас есть отложенная цель — отправляем её
            if self.pending_goal_pose is not None:
                self.get_logger().info('Nav2: sending updated goal after cancel')
                goal = self.pending_goal_pose
                self.pending_goal_pose = None
                self._send_goal_to_nav2(goal)

        else:
            self.get_logger().warn(f'Nav2: goal ended with status {status}')

        # Текущая цель больше не активна (либо завершилась, либо отменена)
        self.nav2_goal_active = False
        self.nav2_goal_handle = None

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    def timer_callback(self):
        # Вращаемся в поиске, если не нашли и не достигли цели
        if not self.target_found and not self.target_reached:
            if self.SAM:
                # В SAM-режиме поворот выполняется внутри обработчика
                return
            self._publish_search_rotation()

    def camera_info_callback(self, msg: CameraInfo):
        if not self.camera_info_received:
            self.fx = msg.k[0]  # fx
            self.fy = msg.k[4]  # fy
            self.cx = msg.k[2]  # cx
            self.cy = msg.k[5]  # cy

            self.camera_info_received = True
            self.get_logger().info(
                f'Camera intrinsics updated: '
                f'fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}'
            )

    def prompt_callback(self, msg: String):
        if self.current_prompt != msg.data:
            self.current_prompt = msg.data
            self.get_logger().info(f'Новый промпт получен: "{self.current_prompt}"')
            self.target_found = False
            self.target_reached = False

    def synced_image_depth_callback(self, rgb_msg: Image, depth_msg: Image):
        """Синхронный callback RGB + depth."""
        if not self.camera_info_received:
            return

        try:
            image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f'Ошибка конвертации изображения: {e}')
            return

        self.latest_depth = depth

        if self.SAM:
            self._handle_sam_mode(image, depth)
        else:
            self._handle_clip_mode(image, depth)

    # -------------------------------------------------------------------------
    # Логика для SAM
    # -------------------------------------------------------------------------
    def _handle_sam_mode(self, image, depth):
        # Нет промпта / depth / цель уже достигнута — нечего делать
        if self.current_prompt is None or depth is None or self.target_reached:
            return

        # Сегментация SAM: возвращает картинку, центр, depth-карту (если нужно) и время
        seg_img, center_coords, image_depth_map, segmentation_time = self.segmentor.segment(
            image,
            self.current_prompt,
            depth,
        )

        self.image_pub.publish(
            self.bridge.cv2_to_imgmsg(seg_img, encoding='bgr8')
        )

        # Объект не найден в кадре
        if center_coords is None:
            if not self.target_found and not self.target_reached:
                self.get_logger().warn('Объект не найден')

                # Поворот в поиске, как раньше
                self._publish_search_rotation()
                time.sleep(4.0)
                self.latest_depth = None
                self.get_logger().info('Ожидание после поворота окончено')
            return

        # Объект найден: первый раз или переобнаружение
        if not self.target_found:
            self.get_logger().info(
                f'[SAM] Координаты центра: ({center_coords[0]}, {center_coords[1]})'
            )
            self.target_found = True

        # Статистика по времени сегментации
        self._add_segmentation_time(segmentation_time)

        # --- Переход: пиксель -> 3D в системе камеры ---
        x_px = int(center_coords[0])
        y_px = int(center_coords[1])

        point_camera = self._pixel_to_camera_point(x_px, y_px, depth)
        if point_camera is None:
            self.get_logger().warn('[SAM] Некорректная глубина в центре объекта')
            return

        if self.target_reached:
            return

        try:
            # --- Переход: камера -> мир (map) ---
            point_world = self._camera_point_to_world(point_camera)

            # Положение робота (base_link) в map
            transform_base = self._lookup_transform(
                'map',
                'base_link',
                timeout_sec=0.5,
            )

            robot_x = transform_base.transform.translation.x
            robot_y = transform_base.transform.translation.y

            dx = point_world.x - robot_x
            dy = point_world.y - robot_y

            distance = np.hypot(dx, dy)

            # Для SAM используем простой отступ: стоять на расстоянии ~offset от объекта
            if distance > self.offset:
                # хотим сместиться на (distance - offset) в сторону объекта
                scale = (distance - self.offset) / distance
            else:
                scale = 0.0
                self.target_reached = True
                self.get_logger().info('[SAM] Объект достигнут')
                self._log_avg_seg_time(
                    'Average GroundingDINO + SAM segmentation time is'
                )

            goal_x = robot_x + dx * scale
            goal_y = robot_y + dy * scale

            self.get_logger().info(
                f'[SAM] Объект в map frame: '
                f'X={point_world.x:.2f}, '
                f'Y={point_world.y:.2f}, '
                f'Z={point_world.z:.2f}'
            )
            self.get_logger().info(
                f'[SAM] Расстояние до цели distance = {distance:.2f}, '
                f'offset = {self.offset:.2f}'
            )

            # Формируем goal для Nav2
            goal = PoseStamped()
            goal.header.frame_id = 'map'
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.pose.position.x = goal_x
            goal.pose.position.y = goal_y

            theta = np.arctan2(dy, dx)
            goal.pose.orientation = self.yaw_to_quaternion(theta)

            # Для SAM, в отличие от CLIP, пока отправляем одну цель (без переобновлений)
            # Если захочешь трекинг как у CLIP — можно завязать на goal_update_period + _update_nav2_goal
            if scale > 0.0 and not self.nav2_goal_active:
                self._send_goal_to_nav2(goal)

        except Exception as e:
            self.get_logger().error(f'Ошибка трансформации в map (SAM): {e}')


    # -------------------------------------------------------------------------
    # Логика для CLIP (SAM = False)
    # -------------------------------------------------------------------------
    def _handle_clip_mode(self, image, depth):
        # Без промпта, depth или если цель уже достигнута – ничего не делаем
        if self.current_prompt is None or depth is None or self.target_reached:
            return

        now = self.get_clock().now()

        # --- Фаза поиска: target_found == False ---
        # В этой фазе работаем на каждый кадр, без ограничений по времени.
        # --- Фаза сопровождения: target_found == True ---
        # В этой фазе ограничиваем частоту обновления цели.
        if self.target_found:
            if (now - self.last_goal_update_time) < Duration(
                seconds=self.goal_update_period
            ):
                return

        # Сегментация:
        seg_img, center_coords, segmentation_time = self.segmentor.segment(
            image,
            self.current_prompt,
        )

        self.image_pub.publish(
            self.bridge.cv2_to_imgmsg(seg_img, encoding='bgr8')
        )

        if center_coords is None:
            # Объект не нашли на этом кадре — в фазе поиска просто ждём далее
            return

        # Если объект только что нашёлся впервые
        if not self.target_found:
            self.get_logger().info(
                f'Координаты центра: ({center_coords[0]}, {center_coords[1]})'
            )
            self.target_found = True

        self._add_segmentation_time(segmentation_time)

        # 3D-координаты по depth в системе камеры
        x_px = int(center_coords[0])
        y_px = int(center_coords[1])

        point_camera = self._pixel_to_camera_point(x_px, y_px, depth)
        if point_camera is None:
            self.get_logger().warn('Некорректная глубина в центре объекта')
            return

        if self.target_reached:
            return

        try:
            # Точка в системе map
            point_world = self._camera_point_to_world(point_camera)

            # Трансформа робота (base_link) в map
            transform_base = self._lookup_transform(
                'map',
                'base_link',
                timeout_sec=0.5,
            )

            robot_x = transform_base.transform.translation.x
            robot_y = transform_base.transform.translation.y

            dx = point_world.x - robot_x
            dy = point_world.y - robot_y

            distance = np.hypot(dx, dy)

            # Как и раньше, подходим не вплотную, а с заданным отступом
            if 0.8 * distance > self.offset + self.offset_delta:
                scale = 0.8
            else:
                scale = 0.0  # цель близко, можно останавливаться рядом

            goal_x = robot_x + dx * scale
            goal_y = robot_y + dy * scale

            self.get_logger().info(
                f'Объект в map frame: '
                f'X={point_world.x:.2f}, '
                f'Y={point_world.y:.2f}, '
                f'Z={point_world.z:.2f}'
            )
            self.get_logger().info(
                f'Расстояние до цели distance = {distance:.2f}, '
                f'offset = {self.offset:.2f} +- {self.offset_delta:.2f}'
            )

            goal = PoseStamped()
            goal.header.frame_id = 'map'
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.pose.position.x = goal_x
            goal.pose.position.y = goal_y

            theta = np.arctan2(dy, dx)
            goal.pose.orientation = self.yaw_to_quaternion(theta)

            # Фиксируем время последнего обновления цели
            self.last_goal_update_time = now

            # Обновляем цель Nav2:
            #   - если цели не было — просто отправится;
            #   - если была — сначала отменится старая, потом отправится новая.
            self._update_nav2_goal(goal)

        except Exception as e:
            self.get_logger().error(f'Ошибка трансформации в map: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = SAMNode()
    rclpy.spin(node)
    rclpy.shutdown()
