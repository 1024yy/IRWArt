import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import config as c
from natsort import natsorted


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class IRWArt_Dataset(Dataset):
    def __init__(self, transforms_=None, mode="train"):

        self.transform = transforms_
        self.mode = mode
        if mode == 'train':
            # train
            self.cover_files = natsorted(sorted(glob.glob(c.TRAIN_COVER_PATH + "\*." + c.format_train)))
            self.secret_files = natsorted(sorted(glob.glob(c.TRAIN_SECRET_PATH + "\*." + c.format_train)))
        else:
            # test
            self.cover_files = natsorted(sorted(glob.glob(c.TEST_COVER_PATH + "\*." + c.format_train)))
            self.secret_files = natsorted(sorted(glob.glob(c.TEST_SECRET_PATH + "\*." + c.format_train)))

    def __getitem__(self, index):

        cover_image = Image.open(self.cover_files[index])
        secret_image = Image.open(self.secret_files[index])
        cover_image = to_rgb(cover_image)
        secret_image = to_rgb(secret_image)
        cover_image = self.transform(cover_image)
        secret_image = self.transform(secret_image)
        return cover_image, secret_image


    def __len__(self):
        if self.mode == 'shuffle':
            return max(len(self.files_cover), len(self.files_secret))

        else:
            return len(self.cover_files)


transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.Resize([224,224]),
    T.RandomCrop(c.cropsize),
    T.ToTensor()
])

transform_val = T.Compose([
    T.CenterCrop(c.cropsize_val),
    T.ToTensor(),
])


# Training data loader
trainloader = DataLoader(
    IRWArt_Dataset(transforms_=transform, mode="train"),
    batch_size=c.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=0,
    drop_last=True
)
# Test data loader
testloader = DataLoader(
    IRWArt_Dataset(transforms_=transform_val, mode="val"),
    batch_size=c.batchsize_val,
    shuffle=False,
    pin_memory=True,
    num_workers=0,
    drop_last=False
)