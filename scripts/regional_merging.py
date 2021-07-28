from glob import glob
from scipy.signal import find_peaks
from tqdm import tqdm
import os
import numpy as np
import cv2
import torch
import torch.nn as nn

unet_model = "UNet.pth"

def smooth(x, window_len=11, window='hanning'):
#Ref: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    if x.ndim != 1:
        raise ValueError

    if x.size < window_len:
        raise ValueError


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
#print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def roi(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    
    hist_smooth = smooth(hist.ravel(), 5)
    
    peaks, _ = find_peaks(hist_smooth, distance=10)
    
    threshold = np.std(img[peaks])
    
    binary_mask = (img > threshold)
      
    binary_mask = binary_mask.astype("uint8")
    
    binary_mask = np.where(binary_mask>0, 1, 0)
    
    return binary_mask

def extract_patches(image):
    patches = []
    patch_size = stride = 48
    for j in range(0, image.shape[1] - patch_size + 1, stride):
        for i in range(0, image.shape[0] - patch_size + 1, stride):
            patch = image[i: i+patch_size, j:j+patch_size]
            patches.append(patch)
    return np.array(patches)

def reconstruct_image(patches, image):
    re_image = np.zeros(image.shape)
    sum_mat = np.zeros(image.shape)
    
    patch_size = stride = 48
    index_x = 0
    index_y = 0
    
    for p in range(patches.shape[0]):
        patch = patches[p]
        re_image[index_x:index_x + patch_size, index_y:index_y + patch_size]+=patch
        sum_mat[index_x:index_x + patch_size, index_y:index_y + patch_size]+=1
        index_x+=stride
        if index_x+patch_size>image.shape[0]:
            index_x=0
            index_y+=stride
    return re_image/sum_mat

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img

# define UNET
class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 stride):
        super(BaseConv, self).__init__()

        self.act = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding,
                               stride)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               padding, stride)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 stride):
        super(DownConv, self).__init__()

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv_block = BaseConv(in_channels, out_channels, kernel_size,
                                   padding, stride)

    def forward(self, x):
        x = self.pool1(x)
        x = self.conv_block(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, in_channels_skip, out_channels,
                 kernel_size, padding, stride):
        super(UpConv, self).__init__()

        self.conv_trans1 = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, padding=0, stride=2)
        self.conv_block = BaseConv(
            in_channels=in_channels + in_channels_skip,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride)

    def forward(self, x, x_skip):
        x = self.conv_trans1(x)
        x = torch.cat((x, x_skip), dim=1)
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_class, kernel_size,
                 padding, stride):
        super(UNet, self).__init__()

        self.init_conv = BaseConv(in_channels, out_channels, kernel_size,
                                  padding, stride)

        self.down1 = DownConv(out_channels, 2 * out_channels, kernel_size,
                              padding, stride)

        self.down2 = DownConv(2 * out_channels, 4 * out_channels, kernel_size,
                              padding, stride)

        self.down3 = DownConv(4 * out_channels, 8 * out_channels, kernel_size,
                              padding, stride)

        self.up3 = UpConv(8 * out_channels, 4 * out_channels, 4 * out_channels,
                          kernel_size, padding, stride)

        self.up2 = UpConv(4 * out_channels, 2 * out_channels, 2 * out_channels,
                          kernel_size, padding, stride)

        self.up1 = UpConv(2 * out_channels, out_channels, out_channels,
                          kernel_size, padding, stride)

        self.out = nn.Conv2d(out_channels, n_class, kernel_size, padding, stride)

    def forward(self, x):
        # Encoder
        x = self.init_conv(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        # Decoder
        x_up = self.up3(x3, x2)
        x_up = self.up2(x_up, x1)
        x_up = self.up1(x_up, x)
        x_out = self.out(x_up)
        return x_out

unet = UNet(in_channels=1,
             out_channels=64,
             n_class=1,
             kernel_size=3,
             padding=1,
             stride=1)

unet.load_state_dict(torch.load(unet_model, map_location="cpu"))

unet = unet.to("cpu")

def preprocess(image_path, image_dir):
    
    image_name = image_path.split("/")[2].split(".")[0]
    
    image = cv2.imread(f"{image_path}")
    
    image = cv2.resize(image, (480, 480), cv2.INTER_AREA)
    
    _, G, _ = cv2.split(image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    enh_img = clahe.apply(G)
    
    mask = roi(G)
    
    sub_image = enh_img * mask
    
    sub_image = sub_image / 255.0
    
    patches = extract_patches(sub_image)
    
    patches = torch.from_numpy(patches)
    
    patches = torch.tensor(patches, dtype=torch.float32)
    
    patches.unsqueeze_(1)
    
    with torch.no_grad():
        images = patches.to(device="cpu")
        pred = unet(images)
        pred = torch.sigmoid(pred)
        pred.squeeze_()
    
    re_img = reconstruct_image(pred.numpy(), enh_img)
    
    vessel = np.array(re_img * 255, dtype=np.uint8)
    vessel = np.where(vessel > 5, 255, 0)
    vessel = np.array(vessel, dtype=np.uint8)
    
    image = cv2.add(vessel, enh_img)
    
    image = crop_image_from_gray(image)
    
    image = cv2.resize(image, (480, 480), cv2.INTER_AREA)

    cv2.imwrite(f"{image_dir}/{image_name}.jpg", image)
    
if __name__ == "__main__":
    for img in tqdm(glob("org/proliferative_dr/*.png")):
        preprocess(img, "enc/proliferative_dr")
        # break