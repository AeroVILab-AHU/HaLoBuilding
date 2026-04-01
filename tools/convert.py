import os
import numpy as np
from PIL import Image

# 颜色映射表,将单通道标签对应转换为3通道标签
color_map = [
    [0, 0, 0],        # background  0
    [255,255, 255],   # Building 3

]    ####换

def grayscale_to_rgb(image_array):
    """
    将灰度图像数组转换为 RGB 图像数组
    """
    # 创建一个空的 RGB 图像数组
    height, width = image_array.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    # 遍历每个像素，根据灰度值映射到对应的 RGB 颜色
    for i in range(height):
        for j in range(width):
            pixel_value = image_array[i, j]
            if pixel_value < len(color_map):
                rgb_image[i, j] = color_map[pixel_value]
            else:
                rgb_image[i, j] = [0, 0, 0]  # 如果灰度值超出范围，设置为背景色

    return rgb_image

def convert_folder_to_rgb(input_folder, output_folder):
    """
    将输入文件夹中的灰度图转换为 RGB 图并保存到输出文件夹
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for file_name in os.listdir(input_folder):
        # 构建文件路径
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        # 打开灰度图像
        grayscale_image = Image.open(input_path)
        grayscale_array = np.array(grayscale_image)

        # 将灰度图转换为 RGB 图
        rgb_array = grayscale_to_rgb(grayscale_array)
        rgb_image = Image.fromarray(rgb_array)

        # 保存 RGB 图像
        rgb_image.save(output_path)
        print(f"已转换并保存: {output_path}")

# 使用示例
input_folder = r'/mnt/BigData/sff/LWGANet-main/segmentation/result_bIou/FAM_Dense/lowlight_100e_best_d4'  # 灰度图文件夹路径
output_folder = r'/mnt/BigData/sff/LWGANet-main/segmentation/result_bIou/FAM_Dense/lowlight_100e_best_d4'       # RGB 图输出文件夹路径

convert_folder_to_rgb(input_folder, output_folder)