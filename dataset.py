import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

def convert_to_jpg(folder):
    for filename in os.listdir(folder):
        if filename.lower().endswith('.png'):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert("RGB")
            new_name = os.path.splitext(filename)[0] + ".jpg"
            img.save(os.path.join(folder, new_name), "JPEG")
            os.remove(img_path)


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path_jpg = os.path.join(self.mask_dir, img_name)
        mask_path_png = os.path.join(self.mask_dir, os.path.splitext(img_name)[0] + ".png")
        if os.path.exists(mask_path_jpg):
            mask_path = mask_path_jpg
        elif os.path.exists(mask_path_png):
            mask_path = mask_path_png
        else:
            raise FileNotFoundError(f"No mask found for {img_name}")
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        # 强制统一尺寸
        resize = transforms.Resize((256, 256))
        image = resize(image)
        mask = resize(mask)
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
        return image, mask

# # 示例用法
# if __name__ == "__main__":
#     image_dir = "dataset/train/images"
#     mask_dir = "dataset/train/masks"
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#     ])
#     dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
#     for images, masks in dataloader:
#         print(images.shape, masks.shape)
#         break