#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KDTree

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2


# =================================================
# U-Net Model Definition
# =================================================

class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class BottleneckSpatialAttention(nn.Module):
    """Full spatial self-attention over H x W tokens"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_norm = self.norm(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_out = x_flat + attn_out
        x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x_out


class UNetDenoiser(nn.Module):
    def __init__(
        self,
        in_channels=7,
        num_classes=2,
        base_channels=64,
        bilinear=False,
        attn_heads=4,
    ):
        super().__init__()
        assert in_channels == 7, "Expected 7 input channels"
        
        # Branch encoders
        self.inc_spatial = DoubleConv(5, base_channels // 2)
        self.inc_temporal = DoubleConv(2, base_channels // 2)
        self.inc_fuse = DoubleConv(base_channels, base_channels)
        
        # Encoder
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        
        # Bottleneck Attention
        self.bottleneck_attn = BottleneckSpatialAttention(
            base_channels * 16 // factor,
            num_heads=attn_heads
        )
        
        # Decoder
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = OutConv(base_channels, num_classes)

    def forward(self, x):
        x_spatial = x[:, :5, :, :]
        x_temporal = x[:, 5:, :, :]
        
        f_spatial = self.inc_spatial(x_spatial)
        f_temporal = self.inc_temporal(x_temporal)
        
        x1 = torch.cat([f_spatial, f_temporal], dim=1)
        x1 = self.inc_fuse(x1)
        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x5 = self.bottleneck_attn(x5)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        return self.outc(x)


# =================================================
# Range Image Projection
# =================================================

class VelodyneRangeProjection:
    """Range image projection for Velodyne HDL-64E"""
    def __init__(self, proj_H=64, proj_W=1024, fov_up=2.0, fov_down=-24.9):
        self.proj_H = proj_H
        self.proj_W = proj_W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down

    def project(self, points, remissions):
        """
        Project 3D points to range image
        
        Args:
            points: (N, 3) array of x, y, z coordinates
            remissions: (N,) array of intensity values
            
        Returns:
            proj_range, proj_xyz, proj_remission, proj_mask, proj_idx
        """
        proj_range = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        proj_xyz = np.zeros((self.proj_H, self.proj_W, 3), dtype=np.float32)
        proj_remission = np.zeros((self.proj_H, self.proj_W), dtype=np.float32)
        proj_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)
        
        fov_up = self.proj_fov_up / 180.0 * np.pi
        fov_down = self.proj_fov_down / 180.0 * np.pi
        fov = abs(fov_down) + abs(fov_up)
        
        depth = np.linalg.norm(points, axis=1)
        valid = depth > 0
        depth = depth[valid]
        points = points[valid]
        remission = remissions[valid]
        
        if len(depth) == 0:
            proj_mask = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)
            return proj_range, proj_xyz, proj_remission, proj_mask, proj_idx
        
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]
        
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(np.clip(scan_z / depth, -1.0, 1.0))
        
        proj_x = 0.5 * (yaw / np.pi + 1.0)
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov
        
        proj_x *= self.proj_W
        proj_y *= self.proj_H
        
        proj_x = np.floor(proj_x).astype(np.int32)
        proj_y = np.floor(proj_y).astype(np.int32)
        
        proj_x = np.clip(proj_x, 0, self.proj_W - 1)
        proj_y = np.clip(proj_y, 0, self.proj_H - 1)
        
        indices = np.arange(len(depth))
        order = np.argsort(depth)[::-1]
        
        depth = depth[order]
        points = points[order]
        remission = remission[order]
        proj_x = proj_x[order]
        proj_y = proj_y[order]
        indices = indices[order]
        
        proj_range[proj_y, proj_x] = depth
        proj_xyz[proj_y, proj_x] = points
        proj_remission[proj_y, proj_x] = remission
        proj_idx[proj_y, proj_x] = indices
        
        proj_mask = (proj_idx >= 0).astype(np.int32)
        
        return proj_range, proj_xyz, proj_remission, proj_mask, proj_idx


# =================================================
# Back Projection
# =================================================

class RangeImageBackProjection:
    """Back-project range image predictions to 3D point cloud"""
    def __init__(self, proj_H=64, proj_W=1024):
        self.proj_H = proj_H
        self.proj_W = proj_W

    def back_project_with_knn(
        self, pred_img, proj_xyz, proj_idx, proj_mask, original_points, k=5
    ):
        """
        Back-project predictions using KNN for unmapped points
        
        Returns:
            point_labels: (N,) predicted labels (0=noise, 1=clean)
        """
        num_points = len(original_points)
        point_labels = np.ones(num_points, dtype=np.int32)
        mapped_mask = np.zeros(num_points, dtype=bool)
        
        # Direct mapping
        for i in range(self.proj_H):
            for j in range(self.proj_W):
                if proj_mask[i, j] > 0:
                    point_idx = proj_idx[i, j]
                    if 0 <= point_idx < num_points:
                        point_labels[point_idx] = pred_img[i, j]
                        mapped_mask[point_idx] = True
        
        # KNN for unmapped points
        unmapped_indices = np.where(~mapped_mask)[0]
        
        if len(unmapped_indices) > 0:
            mapped_indices = np.where(mapped_mask)[0]
            
            if len(mapped_indices) > 0:
                mapped_points = original_points[mapped_indices]
                mapped_labels = point_labels[mapped_indices]
                
                kdtree = KDTree(mapped_points)
                unmapped_points = original_points[unmapped_indices]
                distances, indices = kdtree.query(
                    unmapped_points, k=min(k, len(mapped_points))
                )
                
                for i, unmapped_idx in enumerate(unmapped_indices):
                    neighbor_labels = mapped_labels[indices[i]]
                    point_labels[unmapped_idx] = np.bincount(neighbor_labels).argmax()
        
        return point_labels


# =================================================
# ROS2 Node
# =================================================

class WeatherNoiseFilterNode(Node):
    def __init__(self):
        super().__init__('weather_noise_filter')
        
        # Parameters
        self.declare_parameter('model_path', '/home/kbkn')
        self.declare_parameter('proj_h', 64)
        self.declare_parameter('proj_w', 1024)
        self.declare_parameter('fov_up', 2.0)
        self.declare_parameter('fov_down', -24.9)
        self.declare_parameter('base_channels', 64)
        self.declare_parameter('knn_k', 5)
        self.declare_parameter('use_temporal', True)
        self.declare_parameter('input_topic', '/points_raw')
        self.declare_parameter('output_topic', '/points_filtered')
        
        model_path = self.get_parameter('model_path').value
        self.proj_h = self.get_parameter('proj_h').value
        self.proj_w = self.get_parameter('proj_w').value
        self.fov_up = self.get_parameter('fov_up').value
        self.fov_down = self.get_parameter('fov_down').value
        base_channels = self.get_parameter('base_channels').value
        self.knn_k = self.get_parameter('knn_k').value
        self.use_temporal = self.get_parameter('use_temporal').value
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')
        
        # Load model
        self.get_logger().info('Loading model...')
        self.model = UNetDenoiser(
            in_channels=7,
            num_classes=2,
            base_channels=base_channels,
            bilinear=False,
            attn_heads=4
        ).to(self.device)
        
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.get_logger().info(f'Loaded model from {model_path}')
        else:
            self.get_logger().warn('No model path provided, using untrained model')
        
        self.model.eval()
        
        # Range projection/back-projection
        self.projector = VelodyneRangeProjection(
            self.proj_h, self.proj_w, self.fov_up, self.fov_down
        )
        self.back_projector = RangeImageBackProjection(self.proj_h, self.proj_w)
        
        # Previous frame for temporal features
        self.prev_frame = None
        
        # ROS2 pub/sub
        self.subscription = self.create_subscription(
            PointCloud2,
            input_topic,
            self.pointcloud_callback,
            10
        )
        self.publisher = self.create_publisher(PointCloud2, output_topic, 10)
        
        self.get_logger().info('Weather noise filter node initialized')
        self.get_logger().info(f'Subscribing to: {input_topic}')
        self.get_logger().info(f'Publishing to: {output_topic}')
        self.get_logger().info(f'Temporal features: {"enabled" if self.use_temporal else "disabled"}')

    def pointcloud_callback(self, msg):
        try:
            # Convert PointCloud2 to numpy
            points_list = []
            for point in pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
                points_list.append(point)
            
            if len(points_list) == 0:
                self.get_logger().warn('Received empty point cloud')
                return
            
            points_array = np.array(points_list, dtype=np.float32)
            xyz = points_array[:, :3]
            intensity = points_array[:, 3] / 255.0  # Normalize
            
            # Project to range image
            proj_range, proj_xyz, proj_intensity, proj_mask, proj_idx = \
                self.projector.project(xyz, intensity)
            
            # Initialize temporal difference channels
            delta_range = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)
            delta_intensity = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)
            
            # Compute temporal differences
            if self.use_temporal and self.prev_frame is not None:
                prev_range, prev_intensity_proj, prev_mask = self.prev_frame
                valid_both = (proj_mask > 0) & (prev_mask > 0)
                
                if valid_both.sum() > 0:
                    delta_range[valid_both] = proj_range[valid_both] - prev_range[valid_both]
                    delta_intensity[valid_both] = proj_intensity[valid_both] - prev_intensity_proj[valid_both]
            
            # Store current frame for next iteration
            if self.use_temporal:
                self.prev_frame = (proj_range.copy(), proj_intensity.copy(), proj_mask.copy())
            
            # Create input tensor: [x, y, z, intensity, range, Î"range, Î"intensity]
            input_img = np.stack([
                proj_xyz[:, :, 0],
                proj_xyz[:, :, 1],
                proj_xyz[:, :, 2],
                proj_intensity,
                proj_range,
                delta_range,
                delta_intensity,
            ], axis=0).astype(np.float32)
            
            # Normalize
            valid_pixels = proj_mask > 0
            for c in range(7):
                channel = input_img[c]
                if valid_pixels.sum() == 0:
                    channel[:] = 0
                    continue
                
                if c in [5, 6]:  # Temporal channels: clip only
                    channel[valid_pixels] = np.clip(channel[valid_pixels], -5.0, 5.0)
                    channel[~valid_pixels] = 0
                else:  # Spatial channels: normalize
                    valid_values = channel[valid_pixels]
                    mean = valid_values.mean()
                    std = valid_values.std() + 1e-6
                    channel[valid_pixels] = (valid_values - mean) / std
                    channel[~valid_pixels] = 0
            
            # Model inference
            input_tensor = torch.from_numpy(input_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                pred_img = torch.argmax(outputs, dim=1)[0].cpu().numpy()
            
            # Back-project to point cloud (0=noise, 1=clean)
            point_labels = self.back_projector.back_project_with_knn(
                pred_img, proj_xyz, proj_idx, proj_mask, xyz, k=self.knn_k
            )
            
            # Filter points (keep only clean points)
            clean_mask = point_labels == 1
            filtered_xyz = xyz[clean_mask]
            filtered_intensity = (intensity[clean_mask] * 255.0).astype(np.float32)
            
            # Publish filtered point cloud
            filtered_msg = self.create_pointcloud2(
                filtered_xyz,
                filtered_intensity,
                msg.header
            )
            self.publisher.publish(filtered_msg)
            
            # Log stats
            noise_count = np.sum(point_labels == 0)
            clean_count = np.sum(point_labels == 1)
            self.get_logger().info(
                f'Filtered: {noise_count} noise, {clean_count} clean '
                f'({noise_count/(noise_count+clean_count)*100:.1f}% removed)',
                throttle_duration_sec=1.0
            )
            
        except Exception as e:
            self.get_logger().error(f'Error processing point cloud: {str(e)}')

    def create_pointcloud2(self, xyz, intensity, header):
        """Create PointCloud2 message from xyz and intensity"""
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        points = np.zeros(len(xyz), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.float32),
        ])
        
        points['x'] = xyz[:, 0]
        points['y'] = xyz[:, 1]
        points['z'] = xyz[:, 2]
        points['intensity'] = intensity
        
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(xyz)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = points.tobytes()
        
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = WeatherNoiseFilterNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()