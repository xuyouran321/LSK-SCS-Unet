import os
import cv2
import numpy as np

# 定义每组图像的大小和行列数
image_size = (1024, 1024)
rows = 2
columns = 4

# 定义文件夹路径
input_folder = "output_img/new_test/a2fpn/patch"
output_folder = "output_img/new_test/a2fpn/concat"

# 创建保存拼接图像的文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def concat_images():
    # 初始化字典来存储每组图像的名称和路径
    image_groups = {}
    # 获取输入文件夹中的所有图像文件并进行排序
    image_files = sorted(os.listdir(input_folder))
    # 遍历图像文件列表并将图像按组存储到字典中
    for image_file in image_files:
        image_name = image_file.split("_")[0] + image_file.split("_")[1]
        if image_name not in image_groups:
            image_groups[image_name] = []
        image_path = os.path.join(input_folder, image_file)
        image_groups[image_name].append(image_path)
    return image_groups

# 定义函数来拼接图像
def concatenate_images(image_paths):
    # 创建一个空白画布来存放拼接后的图像
    canvas = np.zeros((rows * image_size[1], columns * image_size[0], 3), dtype=np.uint8)
    
    # 遍历每张图像的路径和对应的行列位置并进行拼接
    for i, image_path in enumerate(image_paths):
        row = i // columns
        column = i % columns
        image = cv2.imread(image_path)
        canvas[row * image_size[1]:(row + 1) * image_size[1],
               column * image_size[0]:(column + 1) * image_size[0]] = image
        
    return canvas

if __name__ == '__main__':
    # 遍历字典中的每组图像并进行拼接和保存
    image_groups = concat_images()
    for image_name, image_paths in image_groups.items():
        concat_image = concatenate_images(image_paths)
        output_name = image_name + ".png"
        output_path = os.path.join(output_folder, output_name)

        # 将NumPy数组转换回图像格式并保存
        cv2.imwrite(output_path, concat_image)
    print("图像拼接完成！")