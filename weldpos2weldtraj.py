import numpy as np
import open3d as o3d

def interpolate_weldpos(weldpos_workpiece, num_points):
    """
    插值焊接位置坐标
    
    Args:
        weldpos_workpiece (list/ndarray): 原始焊接位置坐标 [[x1,y1,z1], [x2,y2,z2]]
        num_points (int): 插值后的总点数
    
    Returns:
        ndarray: 插值后的焊接位置坐标数组
    """
    # 转换输入为numpy数组以提高效率
    points = np.array(weldpos_workpiece)
    # points = weldpos_workpiece
    # print('points:', points)
    # 生成等间距的x坐标
    x_interp = np.linspace(points[0,0], points[-1,0], num_points)
    
    # 对y和z坐标进行插值
    y_interp = np.interp(x_interp, points[:,0], points[:,1])
    z_interp = np.interp(x_interp, points[:,0], points[:,2])
    
    return np.column_stack((x_interp, y_interp, z_interp))

def load_transforms(filepath="./transforms/transform_params.npy"):
    """
    加载变换参数
    
    Args:
        filepath (str): 变换参数文件路径
        
    Returns:
        dict: 包含rotation_matrix和translation_vector的字典
    """
    try:
        return np.load(filepath, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading transform parameters: {e}")
        return None

def create_point_cloud(points, color):
    """
    创建彩色点云
    
    Args:
        points (ndarray): 点云坐标
        color (list): RGB颜色值[r,g,b]
    
    Returns:
        open3d.geometry.PointCloud: 着色后的点云对象
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    return pcd

def main():
    # 初始焊接位置
    # weldpos_workpiece = np.array([[0, -1.8253968253968254, 5], 
    #                              [100, -1.8253968253968254, 5]])
    length = float(input("请输入焊道长度："))
    weldpos_workpiece = [[0, -1.8253968253968254, 5], 
                                 [length, -1.8253968253968254, 5]]
    
    # 加载变换参数
    params = load_transforms()
    if params is None:
        return
    
    # 生成焊接轨迹
    # 先插值再坐标变换
    weldtraj_workpiece = interpolate_weldpos(weldpos_workpiece, 100)
    weldtraj_camera = np.dot(weldtraj_workpiece, params['rotation_matrix'].T) + params['translation_vector']

    # 先坐标变换再插值
    weldpos_camera = np.dot(weldpos_workpiece, params['rotation_matrix'].T) + params['translation_vector']
    weldtraj_camera2 = interpolate_weldpos(weldpos_camera, 50)
    
    # 创建点云对象
    point_weldpos = create_point_cloud(weldtraj_camera, [0, 1, 0])  # 绿色轨迹
    point_weldpos2 = create_point_cloud(weldtraj_camera2, [0, 0, 1])  # 蓝色轨迹
    
    # 加载工件点云
    pcd = o3d.io.read_point_cloud("./pointcloud/thickboard.pcd")
    
    # 创建坐标系
    frames = [
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=50),  # 相机坐标系
        o3d.geometry.TriangleMesh.create_coordinate_frame(  # 工件坐标系
            size=50, 
            origin=params['translation_vector']
        ).rotate(params['rotation_matrix'], center=params['translation_vector'])
    ]
    
    # 可视化
    o3d.visualization.draw_geometries([pcd, point_weldpos, point_weldpos2] + frames)

if __name__ == "__main__":
    main()