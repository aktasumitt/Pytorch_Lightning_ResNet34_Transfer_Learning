from torchvision.transforms import transforms
from pathlib import Path
from src.exception.exception import ExceptionNetwork, sys

def get_img_label_dict(path,labels):
    try:
        
        img_label_dict={}
        for img_folder in Path(path).glob("*"):
            label=labels[img_folder.name]
            for img in img_folder.glob("*"):
                img_label_dict[img]=label
        return img_label_dict
    
    except Exception as e:
            raise ExceptionNetwork(e, sys)
    
def get_transformer(img_resize_size):
        try:
            # base transforms for all stages
            base_transforms = [transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                transforms.Resize((img_resize_size, img_resize_size))]
            
            # train test ve valid transformer
            train_transformer = transforms.Compose(base_transforms)
            valid_transformer = transforms.Compose(base_transforms)
            test_transformer = transforms.Compose(base_transforms)
            return train_transformer,valid_transformer,test_transformer
        
        except Exception as e:
            raise ExceptionNetwork(e, sys)