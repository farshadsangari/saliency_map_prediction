from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2



def transformer():
    transform_image = transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.Resize((240,320)),
                                            transforms.ToTensor()  ,
                                            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),  
                                        ])

    transform_mask = transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.Resize((236,316)),
                                            transforms.ToTensor()  ,
                                            transforms.Normalize(mean=(0.5), std=(0.5)),  
                                        ])

    return [transform_image,transform_mask]



class myDataset(Dataset):
    def __init__(self,images_directory,Saliency_maps,transform,transform_saliency):
        self.img_files = images_directory
        self.transform = transform
        self.Saliency_maps = Saliency_maps
        self.transform_saliency = transform_saliency

    def __len__(self):
        return len(self.img_files)
        
    def __getitem__(self,index):
        img = cv2.imread(self.img_files[index])
        Saliency_maps_image = cv2.imread(self.Saliency_maps[index],0)
        return self.transform(img),self.transform_saliency(Saliency_maps_image)
    
    
def train_val_loader(train_path ,
                      val_path ,
                      transform_image ,
                      transform_mask,
                      batch_size):
    
    train_images_directory = list(train_path['x_path'].values)
    train_mask_directory = list(train_path['y_path'].values)
    train_data = myDataset(train_images_directory,train_mask_directory,transform_image,transform_mask)


    # val Dataset
    val_images_directory = list(val_path['x_path'].values)
    val_mask_directory = list(val_path['y_path'].values)
    val_data = myDataset(val_images_directory,val_mask_directory,transform_image,transform_mask)


    # Train and val DataLoader
    train_loader = DataLoader(dataset = train_data , batch_size = batch_size,shuffle=True)
    val_loader = DataLoader(dataset = val_data , batch_size =batch_size,shuffle=True)
    return [train_loader,val_loader]
