import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def project_to_yz_plane(pcd, x_crosssection, y1_crop, y2_crop, resolution=20.0, x_threshold=0.1, z_range=19):
    points = np.asarray(pcd.points)
    print("Points dimensions:", points.shape)
    
    mask = (np.abs(points[:, 0]) > x_crosssection) & (np.abs(points[:, 0]) < (x_crosssection + x_threshold))
    yz_points = points[mask][:, 1:]
    
    if len(yz_points) == 0:
        print("没有找到在指定范围内的点！")
        return None
    
    y_min, z_min = np.min(yz_points, axis=0)
    y_max, z_max = np.max(yz_points, axis=0)
    print(f"z范围: {z_min:.2f} to {z_max:.2f}, y范围: {y_min:.2f} to {y_max:.2f}")
    
    height = int(z_range / resolution) + 1
    width = int((y2_crop - y1_crop) / resolution) + 1
    
    image = np.zeros((height, width), dtype=np.uint8)
    
    for point in yz_points:
        y, z = point
        pixel_x = int((y - y1_crop) / resolution)
        pixel_y = int(z / resolution)
        if 0 <= pixel_x < width and 0 <= pixel_y < height:
            image[height - 1 - pixel_y, pixel_x] = 255

    return image

# 将点云数据从camera_frame转换到workpiece_frame
def transform_point_cloud(pcd, rotation_matrix, translation_vector):
    """
    将点云数据从一个坐标系转换到另一个坐标系
    
    参数:
    pcd: Open3D点云对象
    rotation_matrix: 旋转矩阵
    translation_vector: 平移向量
    """
    # 获取点云中的点
    points = np.asarray(pcd.points)
    
    # 应用旋转矩阵和平移向量进行坐标变换
    transformed_points = np.dot(points, rotation_matrix) -np.dot(translation_vector, rotation_matrix)
    # points -= translation_vector  
    # transformed_points = np.dot(points, rotation_matrix) 
    
    # 创建新的点云对象
    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
    
    return transformed_pcd

def main():
    global y1_crop, y2_crop
    
    try:
        # 设置裁剪范围
        y1_crop = -30
        y2_crop = 0

        # 读取和处理点云
        workpiece = "thickboard.pcd"
        pcd = o3d.io.read_point_cloud('./pointcloud/' + workpiece)
        

        plane_model_1, inliers_1 = pcd.segment_plane(distance_threshold=0.5, ransac_n=3, num_iterations=1000)
        inlier_cloud_1 = pcd.select_by_index(inliers_1)
        outlier_cloud_1 = pcd.select_by_index(inliers_1, invert=True)

        # 分割第二个平面
        plane_model_2, inliers_2 = outlier_cloud_1.segment_plane(distance_threshold=0.5, ransac_n=3, num_iterations=1000)
        inlier_cloud_2 = outlier_cloud_1.select_by_index(inliers_2)
        outlier_cloud_2 = outlier_cloud_1.select_by_index(inliers_2, invert=True)

        # 获取平面法向量
        normal_1 = np.array(plane_model_1[:3])
        normal_2 = np.array(plane_model_2[:3])

        # 归一化法向量
        normal_1 = normal_1 / np.linalg.norm(normal_1)
        normal_2 = normal_2 / np.linalg.norm(normal_2)

        # 计算坐标系的旋转矩阵
        # 使用第一个平面的法向量作为z轴
        z_axis = - normal_1
        y_axis = normal_2
        # 计算x轴（使用叉积）
        x_axis = np.cross(normal_1, normal_2)
        x_axis = x_axis / np.linalg.norm(x_axis)



        # 构建旋转矩阵
        rotation_matrix = np.vstack((x_axis, y_axis, z_axis)).T
        # print("旋转矩阵:", rotation_matrix)
        # 计算逆矩阵
        # inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
        # # print("逆矩阵:", inverse_rotation_matrix)

        # 找到最高点作为坐标系原点
        points = np.asarray(pcd.points)
        max_z_point = points[np.argmax(points[:, 2])]
        # print("坐标系原点:", max_z_point)


        # 显示相机坐标系及工件坐标系
        # 创建坐标系
        frames = []
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
        frames.append(camera_frame)
        workpiece_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=max_z_point)
        workpiece_frame.rotate(rotation_matrix, center=max_z_point)
        frames.append(workpiece_frame)

        # 可视化
        o3d.visualization.draw_geometries([inlier_cloud_1, outlier_cloud_1] + [frames[0]])
        # o3d.visualization.draw_geometries([pcd] + frames)
        # o3d.visualization.draw_geometries([pcd, camera_frame])


        ###########################################################
        # 把点云数据从camera_frame转换到workpiece_frame
        ###########################################################
        # 计算平移向量
        translation_vector =   max_z_point
        # translation_vector = [-3.71207905, -0.55968571, 537.33026123]

        # 转换点云数据
        transformed_pcd = transform_point_cloud(pcd, rotation_matrix , translation_vector)

        # 显示转换后的点云数据
        o3d.visualization.draw_geometries([pcd, transformed_pcd, camera_frame])
        # o3d.visualization.draw_geometries([pcd, camera_frame])
        # o3d.visualization.draw_geometries([pcd, transformed_pcd])
        o3d.io.write_point_cloud("./pointcloud/transformed_pcd.pcd", transformed_pcd)
        # print("平面1法向量:", normal_1)
        # print("平面2法向量:", normal_2)
        # print(camera_frame)

        # 创建变换参数字典
        transform_params = {
            'rotation_matrix': rotation_matrix,
            'translation_vector': translation_vector,
            'normal_1': normal_1,
            'normal_2': normal_2,
            'x_axis': x_axis,
            'y_axis': y_axis,
            'z_axis': z_axis
        }

        # 保存变换参数
        if not os.path.exists("./transforms"):
            os.makedirs("./transforms")
            
        np.save("./transforms/transform_params.npy", transform_params)
        
        # 保存为可读文本
        with open("./transforms/transform_params.txt", 'w') as f:
            for key, value in transform_params.items():
                f.write(f"{key}:\n{value}\n\n")


        ###########################################################
        # 切片 投影
        ###########################################################
        # Crop Point Cloud
        # Define a bounding box for cropping
        new_pcd = o3d.io.read_point_cloud("./pointcloud/transformed_pcd.pcd")
        y1_crop = -30
        y2_crop = 0 
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-15, y1_crop, -50), max_bound=(100, y2_crop, 50))

        # Crop the point cloud using the bounding box
        cropped_pcd = new_pcd.crop(bbox)

        # Display the cropped point cloud data

        workpiece_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
        o3d.visualization.draw_geometries([cropped_pcd] + [workpiece_frame])

        # Denoise Point Cloud
        # Apply statistical outlier removal to denoise the point cloud
        cl, ind = cropped_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)

        # Select inliers (denoised point cloud)
        denoised_pcd = cropped_pcd.select_by_index(ind)
        # denoised_pcd = pcd.select_by_index(ind)

        # Display the denoised point cloud data
        o3d.visualization.draw_geometries([denoised_pcd])

        # 确保输出目录存在
        if not os.path.exists("./images/" + workpiece):
            os.makedirs("./images/" + workpiece)

        # 投影并保存图片
        x_crosssection = 8
        img = project_to_yz_plane(denoised_pcd, x_crosssection, y1_crop, y2_crop, resolution=0.05, x_threshold=0.5)
        
        if img is not None:
            img_pil = Image.fromarray(img)
            img_pil.save('./images/' + workpiece + f"/projected_image_{x_crosssection}.png")
            # img_pil.save(f"./images/projected_image_{x_crosssection}.png")
            plt.imshow(img, cmap='gray')
            plt.show()
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == "__main__":
    main()