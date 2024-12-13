import numpy as np
import open3d as o3d

def interpolate_weldpos(weldpos_workpiece, num_points):
    """
    插值 weldpos_workpiece 中的 yz 坐标，z 坐标保持不变，x 坐标等间距
    
    参数:
    weldpos_workpiece: 原始焊接位置坐标列表，格式为 [[x1, y1, z1], [x2, y2, z2]]
    num_points: 插值后的总点数
    
    返回:
    interpolated_points: 插值后的焊接位置坐标列表
    """
    # 提取原始坐标
    x_coords = [pos[0] for pos in weldpos_workpiece]
    y_coords = [pos[1] for pos in weldpos_workpiece]
    z_coords = [pos[2] for pos in weldpos_workpiece]
    
    # 生成等间距的 x 坐标
    x_interp = np.linspace(x_coords[0], x_coords[-1], num_points)
    
    # 进行线性插值
    y_interp = np.interp(x_interp, x_coords, y_coords)
    z_interp = np.interp(x_interp, x_coords, z_coords)  # z 坐标保持不变
    
    # 组合插值后的坐标
    interpolated_points = np.column_stack((x_interp, y_interp, z_interp))
    
    return interpolated_points

def load_transforms():
    """加载保存的变换参数"""
    try:
        transform_params = np.load("./transforms/transform_params.npy", allow_pickle=True).item()
        return transform_params
    except Exception as e:
        print(f"加载变换参数时出错: {e}")
        return None
    
# 示例调用
weldpos_workpiece = [[0, -1.8253968253968254, 5], [100, -1.8253968253968254, 5]]
num_points = 100  # 插值后的总点数

params = load_transforms()
weldtraj_workpiece = interpolate_weldpos(weldpos_workpiece, num_points)
weldtraj_camera = np.dot(weldtraj_workpiece, params['rotation_matrix'].T) + params['translation_vector']
weldpos_camera = np.dot(weldpos_workpiece, params['rotation_matrix'].T) + params['translation_vector']
weldtraj_camera2 = interpolate_weldpos(weldpos_camera, 50)
# print(weldtraj_camera2)

point_weldpos = o3d.geometry.PointCloud()
point_weldpos.points = o3d.utility.Vector3dVector(weldtraj_camera)
point_weldpos.paint_uniform_color([0, 1, 0])  # 将点设置为绿色

point_weldpos2 = o3d.geometry.PointCloud()
point_weldpos2.points = o3d.utility.Vector3dVector(weldtraj_camera2)
point_weldpos2.paint_uniform_color([0, 0, 1])  # 将点设置为蓝色

# 可视化
workpiece = "./pointcloud/thickboard.pcd"
pcd = o3d.io.read_point_cloud(workpiece)

frames = []
camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
frames.append(camera_frame)
workpiece_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=params['translation_vector'])
workpiece_frame.rotate(params['rotation_matrix'], center=params['translation_vector'])
frames.append(workpiece_frame)

# frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
o3d.visualization.draw_geometries([pcd, point_weldpos, point_weldpos2] + frames)