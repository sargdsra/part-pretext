import random
import torchvision
from torch.utils.data import Dataset
from PIL import Image


class PartLoader(Dataset):
    def __init__(self, root_dir, num_parts):
        super(PartLoader, self).__init__()
        self.num_parts = num_parts
        self.root_dir = root_dir
        file_ind = open(self.root_dir)
        self.data = [line.rstrip() for line in file_ind.readlines()]
        file_ind.close()
        self.color_transform = torchvision.transforms.ColorJitter(brightness = 0.4, contrast = 0.4, saturation = 0.4, hue = 0.4)
        self.flips = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomVerticalFlip()]
        self.normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.orig_size = 300
        self.crop_size = 224
        self.sq = int(self.num_parts ** (0.5))
        self.alt = self.orig_size // self.sq
        self.crop_parts = [(i * self.alt, j * self.alt, (i + 1) * self.alt, (j + 1) * self.alt) for i in range(self.sq) for j in range(self.sq)]
        self.res_crop_size = self.crop_size // self.sq

        
    def get_image(self, path):
        image = Image.open(path).convert('RGB')
        return image
    
    def __len__(self):
        return len(self.data)
       
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        original = self.get_image(self.data[index])
        image = torchvision.transforms.Resize((self.orig_size, self.orig_size))(original)
        part = random.choice(list(range(self.num_parts)))
        image_part = image.crop(self.crop_parts[part])
        image_part = torchvision.transforms.RandomCrop((self.res_crop_size, self.res_crop_size))(image_part)
        image_part = torchvision.transforms.Resize((self.res_crop_size, self.res_crop_size))(image_part)
        image_part = self.color_transform(image_part)
        image_part = self.flips[0](image_part)
        image_part = self.flips[1](image_part)
        image_part = torchvision.transforms.functional.to_tensor(image_part)
        image_part = self.normalize(image_part)
        return {'part': image_part, 'part_index': part}