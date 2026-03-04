import os
import random
import shutil

def split_dataset(image_dir, mask_dir, train_image_dir, train_mask_dir, test_image_dir, test_mask_dir, test_ratio=0.2, seed=42):
    random.seed(seed)
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
    images.sort()
    random.shuffle(images)
    test_size = int(len(images) * test_ratio)
    test_images = images[:test_size]
    train_images = images[test_size:]

    # 创建目标文件夹
    for d in [train_image_dir, train_mask_dir, test_image_dir, test_mask_dir]:
        os.makedirs(d, exist_ok=True)

    # 拷贝训练集
    for img_name in train_images:
        shutil.copy(os.path.join(image_dir, img_name), os.path.join(train_image_dir, img_name))
        mask_name = img_name  # 假设掩码文件名和图片一致
        mask_path = os.path.join(mask_dir, mask_name)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(mask_dir, os.path.splitext(mask_name)[0] + ".png")
        shutil.copy(mask_path, os.path.join(train_mask_dir, mask_name))

    # 拷贝测试集
    for img_name in test_images:
        shutil.copy(os.path.join(image_dir, img_name), os.path.join(test_image_dir, img_name))
        mask_name = img_name
        mask_path = os.path.join(mask_dir, mask_name)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(mask_dir, os.path.splitext(mask_name)[0] + ".png")
        shutil.copy(mask_path, os.path.join(test_mask_dir, mask_name))

if __name__ == "__main__":
    split_dataset(
        image_dir="dataset/images",
        mask_dir="dataset/masks",
        train_image_dir="dataset/train/images",
        train_mask_dir="dataset/train/masks",
        test_image_dir="dataset/test/images",
        test_mask_dir="dataset/test/masks",
        test_ratio=0.2  # 20%做测试集
    )