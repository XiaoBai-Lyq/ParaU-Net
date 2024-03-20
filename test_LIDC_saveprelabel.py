import PIL
import torch, torchvision
from torch.utils.data import DataLoader,Dataset
import numpy as np
from torch import nn
from skimage import io
import os
import argparse
from PIL import Image
from utils import AverageMeter
from tqdm import tqdm
from metrics1 import iou_score, dice_coef,precision_coef,recall_coef,calculate_mae,compute_f2_score,compute_f1_2_score,compute_f1_score
from collections import OrderedDict
from Laplacian_Pyramid import laplacian_pyramid,reconstruct_image,reconstruct_image2


#网络模型
from unety import UNett_batcnnorm
from Swin_Unet import vision_transformer
from UNet2P import UNet_2Plus
from UNet3P.models import UNet_3Plus
from unety.SAR_UNet import Se_PPP_ResUNet
from TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from unety import attention_unet
from ResUNet_family import res_unet_plus
import resunet_aspp_up_vit4
from UCTransNet.nets import UCTransNet
import DDANet
import fcn
from MSNet_M2SNet_main.model.miccai_msnet import M2SNet

#my model
from my_model import new_idea

#ablation experiment
from ablation_experiment import UNet_Aspp
from ablation_experiment import UNet_Aspp_SFM
from ablation_experiment.Res2Net import Res2Net_UNet
from ablation_experiment import UNet_Aspp_FCM
from ablation_experiment import Unet_Conformer
from ablation_experiment import Unet_CFFM_Conformer

predimg = []
predimg_color = []
labelimg = []

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="SAR_UNet",
                        help='model name: UNET',choices=['UNET', 'SAR_UNet', "UNet2P",'UNet3P', 'swinunet', 'UCTransNet',"resunetpp"
                                                         'attention_unet', 'TransUNet','resunet_aspp_up_vit4',"DDANet","UNet_vit1_LIDC","FCN",
                                                         "UNet_vit1_LIDC_rnn","new_idea","UNet_Aspp","UNet_Aspp_SFM","M2SNet","Res2Net_UNet","UNet_Aspp_FCM"
                                                         ,"Unet_Conformer","Unet_CFFM_Conformer"])
    config = parser.parse_args()
    return config

def add_alpha_channel(img,fac):
    img = Image.open(img)
    img = img.convert('RGBA')
    # 更改图像透明度
    factor = fac
    img_blender = Image.new('RGBA', img.size, (0, 0, 0, 0))
    img = Image.blend(img_blender, img, factor)
    return img


def image_together(image, layer, save_path, save_name):
    layer = layer
    base = image
    # bands = list(layer.split())
    heigh, width = layer.size
    for i in range(heigh):
        for j in range(width):
            r, g, b, a = layer.getpixel((i, j))
            if r == 0 and g == 0 and b == 0:
                layer.putpixel((i, j), (0, 0, 0, 0))
            if r == 255 and g == 0 and b == 0:
                layer.putpixel((i, j), (255, 0, 0, 0))
            if r == 0 and g == 255 and b == 0:
                layer.putpixel((i, j), (0, 255, 0, 0))
            if r == 0 and g == 0 and b == 255:
                layer.putpixel((i, j), (0, 0, 255, 0))
    base.paste(layer, (0, 0), layer)  # 贴图操作
    base.save(save_path + "/" + save_name)  # 图片保存

class MyData(Dataset):
    def __init__(self, root_dir, label_dir, transformers = None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.image_path = os.listdir(self.root_dir)
        self.label_path = os.listdir(self.label_dir)
        self.transformers = transformers
    def __getitem__(self, idx):  #如果想通过item去获取图片，就要先创建图片地址的一个列表
        img_name = self.image_path[idx]
        label_name = self.label_path[idx]
        img_item_path = os.path.join(self.root_dir, img_name)  # 每个图片的位置
        label_item_path = os.path.join(self.label_dir, label_name)
        image = np.load(img_item_path)

        # image = image * 255
        # image = image.astype(np.uint8)
        # pyramid = laplacian_pyramid(image, 4)
        # reconstructed_image = reconstruct_image(pyramid) / 255

        reconstructed_image = torch.from_numpy(image)
        label = io.imread(label_item_path)/255
        label = torch.from_numpy(label)
        return reconstructed_image,label
    def __len__(self):
        return len(self.image_path)


def testdate(test_loader, model):
    avg_meters = {#'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  "dice":AverageMeter(),
                  'f1-score':AverageMeter(),
                  'MAE':AverageMeter(),
                  'f2-score': AverageMeter(),
                  'f1-2-score': AverageMeter()
    }
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader))
        for input, target in test_loader:
            input = input.float().cuda()
            input = torch.unsqueeze(input,dim=1)
            target = target.long().cuda()
            output = model(input)

            preds = torch.sigmoid(output).cpu()
            preds = (preds > 0.5).float()
            preds = torch.squeeze(preds)
            predimg_color.append(preds)
            predimg.append(preds)
            labelimg.append(torch.squeeze(target.cpu()))

            iou = iou_score(output, target)
            dice = dice_coef(output,target)
            f1_score = compute_f1_score(output,target)
            f2_score = compute_f2_score(output,target)
            f1_2_score = compute_f1_2_score(output, target)
            MAE = calculate_mae(output,target)
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['f1-score'].update(f1_score, input.size(0))
            avg_meters['f2-score'].update(f2_score, input.size(0))
            avg_meters['f1-2-score'].update(f1_2_score, input.size(0))
            avg_meters['MAE'].update(MAE, input.size(0))
            postfix = OrderedDict([
                #('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('f1-score', avg_meters['f1-score'].avg),
                ('f2-score', avg_meters['f2-score'].avg),
                ('f1-2-score', avg_meters['f1-2-score'].avg),
                ('MAE', avg_meters['MAE'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()
    return OrderedDict([#('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('f1-score',avg_meters['f1-score'].avg),
                        ('f2-score', avg_meters['f2-score'].avg),
                        ('f1-2-score', avg_meters['f1-2-score'].avg),
                        ('MAE', avg_meters['MAE'].avg)
    ])


colormap1 = [[0,0,0], [255,255,255]]

def label2image(prelabel,colormap):
    #预测的标签转化为图像，针对一个标签图
    h,w = prelabel.shape
    prelabel = prelabel.reshape(h*w,-1)
    image = np.zeros((h*w,3),dtype=np.uint8)
    for i in range(len(colormap)):
        index = np.where(prelabel == i)
        image[index,:] = colormap[i]
    return image.reshape(h, w, 3)

forecast_label = 'C:\\Users\\beautiful\\Desktop\\lung_keshihua\\forecast_label' #预测标签转换成图片的地址
forecast_label_npy = "C:\\Users\\beautiful\\Desktop\\lung_keshihua\\forecast_label_npy"
labelnpytoimg_dir = 'C:\\Users\\beautiful\\Desktop\\lung_keshihua\\labelnpytoimg_dir' #将labelnpy转换成图片保存的地址
imgnpytoimg_dir = "C:\\Users\\beautiful\\Desktop\\lung_keshihua\\imgnpytoimg_dir"
img_dir = 'testdata/img_npy'  #原始图片npy
label_dir = 'testdata/label'  #原始标签
imgandlabel = "C:\\Users\\beautiful\\Desktop\\lung_keshihua\\imgandlabel"
imgandlabel2 = "C:\\Users\\beautiful\\Desktop\\lung_keshihua\\imgandlabel2"

img_read = os.listdir(img_dir)

dataset = MyData(img_dir, label_dir,transformers=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

config = vars(parse_args())
"""
create model
"""
print("=> creating model")
if config['name'] == "UNET":
    model = UNett_batcnnorm.Unet(1, 1)
elif config['name'] == "swinunet":
    model = vision_transformer.SwinUnet(img_size=128, num_classes=1)
elif config['name'] == "UNet3P":
    model = UNet_3Plus.UNet_3Plus(in_channels=1, n_classes=1)
elif config['name'] == "UNet2P":
    model = UNet_2Plus.Unet2_Plus(num_classes=1)
elif config['name'] == 'SAR_UNet':
    model = Se_PPP_ResUNet(1, 1, deep_supervision=False)
elif config['name'] == 'TransUNet':
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    model = ViT_seg(config_vit, img_size=128, num_classes=1)
elif config['name'] == 'attention_unet':
    model = attention_unet.AttU_Net(1, 1)
elif config['name'] == "resunetpp":
    model = res_unet_plus.ResUnetPlusPlus(1, 1)
elif config['name'] == "resunet_aspp_up_vit4":
    model = resunet_aspp_up_vit4.Unet(1, 1)
elif config['name'] == "UCTransNet":
    model = UCTransNet.UCTransNet(1, 1, img_size=128)
elif config['name'] == "DDANet":
    model = DDANet.CompNet()
elif config['name'] == "FCN":
    model = fcn.FCN(1,1)
elif config['name'] == "new_idea":
    model = new_idea.Unet(1, 1)
elif config['name'] == "UNet_Aspp":
    model = UNet_Aspp.Unet(1, 1)
elif config['name'] == "UNet_Aspp_SFM":
    model = UNet_Aspp_SFM.Unet(1, 1)
elif config['name'] == "M2SNet":
    model = M2SNet()
elif config['name'] == "Res2Net_UNet":
    model = Res2Net_UNet.Unet(1, 1)
elif config['name'] == "UNet_Aspp_FCM":
    model = UNet_Aspp_FCM.Unet(1, 1)
elif config['name'] == "Unet_Conformer":
    model = Unet_Conformer.Unet(1, 1)
elif config['name'] == "Unet_CFFM_Conformer":
    model = Unet_CFFM_Conformer.Unet(1, 1)
else:
    raise ValueError("Wrong Parameters")

model.load_state_dict(torch.load(('checpoint/A_LIDC/{}_base/end_bestmodel_5e-05_CosineAnnealingLR_5_6_7.pth').format(config["name"])))
model.cuda()

if __name__ == "__main__":
    test_log = testdate(test_loader,model)
    print('testdata IOU:{:.4f}, testdata Dice:{:.4f}, testdata f1-score:{:.4f}, testdata f2-score:{:.4f}, testdata f1-2-score:{:.4f}, testdata MAE:{:.4f}'
        .format(test_log['iou'], test_log['dice'], test_log['f1-score'],test_log['f2-score'], test_log['f1-2-score'], test_log["MAE"]))
    for i in range(len(predimg)):
        pre = predimg[i]#预测npy
        preimg2 = label2image(predimg_color[i],colormap=colormap1)#预测label图片
        # preimg2 = predimg_color[i]
        label = label2image(labelimg[i],colormap=colormap1)#原始label图片
        # label = labelimg[i]
        x = np.load(img_dir+"\\"+img_read[i])
        if i < 10:
            test_pre_name = "000{}.png".format(i)
            test_pre_np = "000{}.npy".format(i)

            clip_image_path2 = os.path.join(imgnpytoimg_dir, test_pre_name)
            io.imsave(clip_image_path2, x)
            imgx = add_alpha_channel(clip_image_path2,0.85)
            imgx2 = add_alpha_channel(clip_image_path2,0.85)

            clip_image_path = os.path.join(labelnpytoimg_dir, test_pre_name)
            io.imsave(clip_image_path, label)
            labelx = add_alpha_channel(clip_image_path,0.85)
            image_together(imgx,labelx,imgandlabel,test_pre_name)

            clip_label_path = os.path.join(forecast_label, test_pre_name)
            io.imsave(clip_label_path, preimg2)
            prelabelx = add_alpha_channel(clip_label_path, 0.85)
            image_together(imgx2, prelabelx, imgandlabel2, test_pre_name)

            pre_np = os.path.join(forecast_label_npy, test_pre_np)
            np.save(pre_np, pre)

        if i >= 10 and i < 100:
            test_pre_name = "00{}.png".format(i)
            test_pre_np = "00{}.npy".format(i)

            clip_image_path2 = os.path.join(imgnpytoimg_dir, test_pre_name)
            io.imsave(clip_image_path2, x)
            imgx = add_alpha_channel(clip_image_path2, 0.85)
            imgx2 = add_alpha_channel(clip_image_path2, 0.85)

            clip_image_path = os.path.join(labelnpytoimg_dir, test_pre_name)
            io.imsave(clip_image_path, label)
            labelx = add_alpha_channel(clip_image_path, 0.85)
            image_together(imgx, labelx, imgandlabel, test_pre_name)

            clip_label_path = os.path.join(forecast_label, test_pre_name)
            io.imsave(clip_label_path, preimg2)
            prelabelx = add_alpha_channel(clip_label_path, 0.85)
            image_together(imgx2, prelabelx, imgandlabel2, test_pre_name)

            pre_np = os.path.join(forecast_label_npy, test_pre_np)
            np.save(pre_np, pre)

        if i>=100 and i < 1000:
            test_pre_name = "0{}.png".format(i)
            test_pre_np = "0{}.npy".format(i)

            clip_image_path2 = os.path.join(imgnpytoimg_dir, test_pre_name)
            io.imsave(clip_image_path2, x)
            imgx = add_alpha_channel(clip_image_path2, 0.85)
            imgx2 = add_alpha_channel(clip_image_path2, 0.85)

            clip_image_path = os.path.join(labelnpytoimg_dir, test_pre_name)
            io.imsave(clip_image_path, label)
            labelx = add_alpha_channel(clip_image_path, 0.85)
            image_together(imgx, labelx, imgandlabel, test_pre_name)

            clip_label_path = os.path.join(forecast_label, test_pre_name)
            io.imsave(clip_label_path, preimg2)
            prelabelx = add_alpha_channel(clip_label_path, 0.85)
            image_together(imgx2, prelabelx, imgandlabel2, test_pre_name)

            pre_np = os.path.join(forecast_label_npy, test_pre_np)
            np.save(pre_np, pre)

        if i>=1000:
            test_pre_name = "{}.png".format(i)
            test_pre_np = "{}.npy".format(i)

            clip_image_path2 = os.path.join(imgnpytoimg_dir, test_pre_name)
            io.imsave(clip_image_path2, x)
            imgx = add_alpha_channel(clip_image_path2, 0.85)
            imgx2 = add_alpha_channel(clip_image_path2, 0.85)

            clip_image_path = os.path.join(labelnpytoimg_dir, test_pre_name)
            io.imsave(clip_image_path, label)
            labelx = add_alpha_channel(clip_image_path, 0.85)
            image_together(imgx, labelx, imgandlabel, test_pre_name)

            clip_label_path = os.path.join(forecast_label, test_pre_name)
            io.imsave(clip_label_path, preimg2)
            prelabelx = add_alpha_channel(clip_label_path, 0.85)
            image_together(imgx2, prelabelx, imgandlabel2, test_pre_name)

            pre_np = os.path.join(forecast_label_npy, test_pre_np)
            np.save(pre_np, pre)



