from pathlib import Path
from src.components.data_transformation.dataset import CreateDataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from src.exception.exception import ExceptionNetwork, sys

def load_Data_to_transform(image_paths):
    try:
        image_label_dict = {}
        for image_paths in Path(image_paths).glob("*"):
            image_label_dict[image_paths] = 0
        return image_label_dict
    except Exception as e:
        raise ExceptionNetwork(e, sys)
        

def transform_to_dataloader(image_label_dict, resize_size):
    try:
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Resize((resize_size, resize_size))
        ])
        dataset = CreateDataset(image_label_dict, transformer)
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
        return dataloader
    except Exception as e:
        raise ExceptionNetwork(e, sys)
