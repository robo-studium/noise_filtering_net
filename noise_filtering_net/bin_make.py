#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KDTree
import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution (Depthwise + Pointwise)"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=kernel_size, 
            padding=padding, 
            groups=in_channels, 
            bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class LightDoubleConv(nn.Module):
    """Lightweight (DepthwiseSeparableConv -> BN -> ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.double_conv(x)


class LightDown(nn.Module):
    """Lightweight downscaling: MaxPool + LightDoubleConv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            LightDoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class LightUp(nn.Module):
    """Lightweight upscaling: Bilinear upsample + single DepthwiseSeparableConv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution for classification"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class StudentUNet(nn.Module):
    """Lightweight Student U-Net for Knowledge Distillation"""
    
    def __init__(self, in_channels=5, num_classes=2, base_channels=24):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        
        self.inc = LightDoubleConv(in_channels, base_channels)
        self.down1 = LightDown(base_channels, base_channels * 2)
        self.down2 = LightDown(base_channels * 2, base_channels * 4)
        self.down3 = LightDown(base_channels * 4, base_channels * 8)
        self.down4 = LightDown(base_channels * 8, base_channels * 10)
        
        self.up1 = LightUp(base_channels * 10 + base_channels * 8, base_channels * 8)
        self.up2 = LightUp(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.up3 = LightUp(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up4 = LightUp(base_channels * 2 + base_channels, base_channels)
        
        self.outc = OutConv(base_channels, num_classes)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits


class VelodyneRangeProjection:
    """Range image projection for Velodyne HDL-64E"""
    
    def __init__(self, proj_H=64, proj_W=1024, fov_up=2.0, fov_down=-24.9):
        self.proj_H = proj_H
        self.proj_W = proj_W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down

    def project(self, points, remissions):
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


class RangeImageBackProjection:
    """Back-project range image predictions to 3D point cloud"""
    
    def __init__(self, proj_H=64, proj_W=1024):
        self.proj_H = proj_H
        self.proj_W = proj_W

    def back_project_with_knn(self, pred_img, proj_xyz, proj_idx, proj_mask, original_points, k=5):
        num_points = len(original_points)
        point_labels = np.ones(num_points, dtype=np.int32)
        mapped_mask = np.zeros(num_points, dtype=bool)
        
        for i in range(self.proj_H):
            for j in range(self.proj_W):
                if proj_mask[i, j] > 0:
                    point_idx = proj_idx[i, j]
                    if 0 <= point_idx < num_points:
                        point_labels[point_idx] = pred_img[i, j]
                        mapped_mask[point_idx] = True
        
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


class WeatherNoiseFilterNode(Node):
    def __init__(self):
        super().__init__('weather_noise_filter')
        
        # Parameters
        self.declare_parameter('model_path', '/home/kbkn/simple_pointnet/u_net/outputs_student/best_student_model.pth')
        self.declare_parameter('output_dir', './filtered_pointclouds')
        self.declare_parameter('proj_h', 64)
        self.declare_parameter('proj_w', 1024)
        self.declare_parameter('fov_up', 2.0)
        self.declare_parameter('fov_down', -24.9)
        self.declare_parameter('base_channels', 24)
        self.declare_parameter('knn_k', 5)
        self.declare_parameter('input_topic', '/points_raw')
        
        model_path = self.get_parameter('model_path').value
        self.output_dir = self.get_parameter('output_dir').value
        self.proj_h = self.get_parameter('proj_h').value
        self.proj_w = self.get_parameter('proj_w').value
        self.fov_up = self.get_parameter('fov_up').value
        self.fov_down = self.get_parameter('fov_down').value
        base_channels = self.get_parameter('base_channels').value
        self.knn_k = self.get_parameter('knn_k').value
        input_topic = self.get_parameter('input_topic').value
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        self.get_logger().info(f'Output directory: {self.output_dir}')
        
        # Frame counter
        self.frame_count = 0
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')
        
        # Load model
        self.get_logger().info('Loading model...')
        self.model = StudentUNet(
            in_channels=5,
            num_classes=2,
            base_channels=base_channels,
        ).to(self.device)
        
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
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
        
        # ROS2 subscription
        self.subscription = self.create_subscription(
            PointCloud2,
            input_topic,
            self.pointcloud_callback,
            10
        )
        
        self.get_logger().info('Weather noise filter node initialized')
        self.get_logger().info(f'Subscribing to: {input_topic}')

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
            
            # Create input tensor
            input_img = np.stack([
                proj_xyz[:, :, 0],
                proj_xyz[:, :, 1],
                proj_xyz[:, :, 2],
                proj_intensity,
                proj_range,
            ], axis=0).astype(np.float32)
            
            # Normalize
            valid_pixels = proj_mask > 0
            for c in range(5):
                channel = input_img[c]
                if valid_pixels.sum() == 0:
                    channel[:] = 0
                    continue
                
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
            
            # Back-project to point cloud
            point_labels = self.back_projector.back_project_with_knn(
                pred_img, proj_xyz, proj_idx, proj_mask, xyz, k=self.knn_k
            )
            
            # Filter points (keep only clean points)
            clean_mask = point_labels == 1
            filtered_xyz = xyz[clean_mask]
            filtered_intensity = points_array[clean_mask, 3:4]  # Keep original intensity
            
            # Combine xyz and intensity: (N, 4)
            filtered_points = np.hstack([filtered_xyz, filtered_intensity]).astype(np.float32)
            
            # Save as .bin file
            output_filename = os.path.join(self.output_dir, f'{self.frame_count:06d}.bin')
            filtered_points.tofile(output_filename)
            
            # Log stats
            noise_count = np.sum(point_labels == 0)
            clean_count = np.sum(point_labels == 1)
            self.get_logger().info(
                f'Frame {self.frame_count}: Saved {clean_count} points to {output_filename} '
                f'(removed {noise_count} noise points, {noise_count/(noise_count+clean_count)*100:.1f}%)',
                throttle_duration_sec=1.0
            )
            
            self.frame_count += 1
            
        except Exception as e:
            self.get_logger().error(f'Error processing point cloud: {str(e)}')


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