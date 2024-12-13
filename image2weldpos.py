import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from math import pi
from PIL import Image

def modify_points(points, x1_pix):
    """修改点坐标，将x >= x1_pix的点的x坐标设为x1_pix"""
    col_corner = float('inf')
    for i, point in enumerate(points):
        x, y = point
        if x >= x1_pix:
            col_corner = min(col_corner, y)
            points[i][0] = x1_pix
    return points, col_corner

def load_and_process_image(img_path):
    """加载图像并计算比例尺"""
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    pix_height = image.shape[0]
    pix_width = image.shape[1]
    real_height = 19
    scale = pix_height / real_height
    
    print('image.shape:', image.shape)
    print('scale:', scale)
    
    return image, scale

def extract_and_fit_points(image, modify_points=False, x1_pix=0, col_corner=float('inf')):
    """提取白色点并拟合样条曲线"""
    points = np.column_stack(np.where(image == 255))
    points = points[points[:, 1].argsort()]
    if modify_points:
        for i, point in enumerate(points):
            x, y = point
            if x >= x1_pix:
                col_corner = min(col_corner, y)
                # colum.append(z)
                points[i][0] = x1_pix
        
    # 对相同x坐标的点取平均值
    unique_x = np.unique(points[:, 1])
    avg_points = []
    for x in unique_x:
        y_values = points[points[:, 1] == x][:, 0]
        avg_y = np.mean(y_values)
        avg_points.append([avg_y, x])
    
    avg_points = np.array(avg_points)
    
    # 拟合样条曲线
    x_data = avg_points[:, 1]
    y_data = avg_points[:, 0]
    spline = UnivariateSpline(x_data, y_data, s=0.1)
    
    return points, avg_points, spline

def create_mask(image, x_fit, y_fit):
    """创建并填充掩码"""
    mask = np.zeros_like(image, dtype=np.uint8)
    
    # 绘制拟合曲线
    for i in range(len(x_fit) - 1):
        x1, y1 = int(x_fit[i]), int(y_fit[i])
        x2, y2 = int(x_fit[i + 1]), int(y_fit[i + 1])
        cv2.line(mask, (x1, y1), (x2, y2), 255, 1)
    
    # 填充曲线以下区域
    for x in range(mask.shape[1]):
        y_values = np.where(mask[:, x] == 255)[0]
        if len(y_values) > 0:
            y_min = y_values.min()
            mask[y_min:, x] = 255
            
    return mask

def calculate_weld_position(points, image, scale, z1_weldpos):
    """计算焊接位置"""
    x1_pix = image.shape[0] - scale * z1_weldpos
    col_corner = float('inf')
    
    for i, point in enumerate(points):
        x, y = point
        if x >= x1_pix:
            col_corner = min(col_corner, y)
            points[i][0] = x1_pix
            
    y1_pix = (image.shape[1] + col_corner) / 2
    y1_weldpos = -(image.shape[1] - y1_pix) / scale
    
    return x1_pix, y1_pix, y1_weldpos

def calculate_v_speed(area, v_wire=12, d=2):
    """计算焊接速度"""
    s = pi * (d / 2) ** 2
    v_speed = (s * v_wire * 1000 / 60) / area
    return v_speed

def plot_results(image, x_fit, y_fit, x_data, y_data, mask, area, scale, title):
    """绘制结果"""
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.plot(x_fit, y_fit, color='red', linewidth=1)
    plt.scatter(x_data, y_data, color='blue', s=1)
    plt.imshow(mask, cmap='gray', alpha=0.5)
    plt.title(f'{title}: {area / (scale * scale):.2f} mm^2')
    plt.show()

def main():
    # 加载图像
    workpiece = "thickboard.pcd"
    img_path = './images/' + workpiece + '/projected_image_8.png'
    image, scale = load_and_process_image(img_path)
    
    # 初始轮廓处理
    points, avg_points, spline = extract_and_fit_points(image, modify_points=False)
    x_fit = np.linspace(avg_points[:, 1].min(), avg_points[:, 1].max(), 500)
    y_fit = spline(x_fit)
    mask_0 = create_mask(image, x_fit, y_fit)
    area_0 = np.sum(mask_0 == 255)
    print(f'area_0: {area_0} pix, {area_0 / scale ** 2:.2f} mm^2')
    
    plot_results(image, x_fit, y_fit, avg_points[:, 1], avg_points[:, 0], 
                mask_0, 0, scale, "Initial weld seam")

    # 计算焊接位置
    z1_weldpos = 5
    x1_pix, y1_pix, y1_weldpos = calculate_weld_position(points, image, scale, z1_weldpos)
    print(f'weldpos:(y, z) = ({y1_weldpos:.2f}, {z1_weldpos}) mm')
    
    # 处理第一道焊缝
    # points, col_corner = modify_points(points, x1_pix)  # 先修改points
    # print(points)
    # points = np.array(points)
    # image = Image.fromarray(points)

    points, avg_points, spline = extract_and_fit_points(image, modify_points=True, x1_pix=x1_pix)
    # print(points)
    x_fit = np.linspace(avg_points[:, 1].min(), avg_points[:, 1].max(), 500)
    y_fit = spline(x_fit)
    mask_1 = create_mask(image, x_fit, y_fit)
    mask_1 = mask_1 - mask_0
    area_1 = np.sum(mask_1 == 255)
    print(f'area_1: {area_1} pix, {area_1 / scale ** 2:.2f} mm^2')
    
    # 计算焊接速度
    area_1_mm2 = area_1 / (scale ** 2)
    v_speed = calculate_v_speed(area_1_mm2)
    print(f'v_speed: {v_speed:.2f} mm/s')
    
    # 绘制结果

    plot_results(image, x_fit, y_fit, avg_points[:, 1], avg_points[:, 0], 
                mask_1, area_1, scale, "First weld seam")
    


if __name__ == "__main__":
    main()