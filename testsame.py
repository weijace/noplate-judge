#测试精简代码，即使用训练好的模型，计算相同车辆成对的图片距离

import torch
from yacs.config import CfgNode
from vehicle_reid_pytorch.data import make_basic_dataset, make_test_dataset
from torch.utils.data import DataLoader
from model import ParsingReidModel
from logzero import logger
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
import cv2
import os
import shutil
import time
from tool import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
from torch.nn import functional as F
import albumentations as albu

os.environ['CUDA_VISIBLE_DEVICE'] = "2"

def make_config():
    cfg = CfgNode()
    cfg.desc = "" #对本次实验的简单描述，用于为tensorboard命名
    cfg.stage = "train" #train or eval or test
    cfg.device = "cuda:2"
    cfg.device_ids = "" #if not set, use all gpus

    cfg.model = CfgNode()
    cfg.model.name = "resnet50"
    #if it is set to empty, we will download it from torchvision official website.
    cfg.model.pretrain_path = ""
    cfg.model.last_stride = 1
    cfg.model.neck = "bnneck"
    cfg.model.neck_feat = "after"
    cfg.model.pretrain_choice = "imagenet"
    cfg.model.ckpt_period = 10

    cfg.test = CfgNode()
    cfg.test.feat_norm = True
    cfg.test.remove_junk = True
    cfg.test.period = 10
    cfg.test.device = "cuda:2"
    cfg.test.model_path = "../outputs/test/model_120.pth"
    # split: When the CUDA memory is not sufficient,
    # we can split the dataset into different parts
    # for the computing of distance.
    cfg.test.split = 0
    cfg.test.pairs_path = "/share_data/PVEN-master/samecar"
    cfg.test.q_dir = "/share_data/PVEN-master/sip_data/image_query"
    cfg.test.g_dir = "/share_data/PVEN-master/sip_data/image_test"

    cfg.logging = CfgNode()
    cfg.logging.level = "info"
    cfg.logging.period = 20

    cfg.masks = CfgNode()
    cfg.masks.ENCODER = "se_resnext50_32X4d"
    cfg.masks.ENCODER_WEIGHTs = "imagenet"
    cfg.masks.model_path = "/share_data/PVEN-master/examples/parsing/best_model_trainval.pth"

    return cfg

#calculate cosine distance
def cosine_distance(mat1, mat2):
    mat1_mat2 = np.dot(mat1, mat2.t())
    mat1_norm = np.sqrt(np.multiply(mat1, mat1).sum(axis=1))
    mat1_norm = mat1_norm[:, np.newaxis]
    mat2_norm = np.sqrt(np.multiply(mat2, mat2).sum(axis=1))
    mat2_norm = mat2_norm[:, np.newaxis]
    cosine_dis = np.divide(mat1_mat2, np.dot(mat1_norm, mat2_norm.t()))
    return cosine_dis

def build_model(cfg, num_classes):
    model = ParsingReidModel(num_classes, cfg.model.last_stride, cfg.model.pretrain_path, cfg.model.neck,
                             cfg.model.neck_feat, cfg.model.name, cfg.model.pretrain_choice)
    return model

def main():
    cfg = make_config()

    #加载模型放外面
    model = build_model(cfg, 1).to(0)

    state_dict = torch.load(cfg.test.model_path)

    #Remove the classifier
    remove_keyes = []

    for key, value in state_dict.items():
        if 'classifier' in key:
            remove_keys.append(key)
    for key in remove_keyes:
        del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    logger.info(f"Load model {cfg.test.model_path}")
    #在这里将成对的图片，一张放到query中，一张放入test中
    allplateList = os.listdir(cfg.test.pairs_path)
    for ap in tqdm(allplateList):
        platedir = os.path.join(cfg.test.pairs_path, ap)
        platesList = os.listdir(platedir)  #不同视频
        for p in platesList:
            picsdir = os.path.join(platedir, p)
            picsList = os.listdir(picsdir)
            pic1_name = picsList.pop(0)
            pic2_name = picsList.pop(0)
            pic1_dir = os.path.join(picsdir, pic1_name)
            pic2_dir = os.path.join(picsdir, pic2_name)
            shutil.copy(pic1_dir, cfg.test.q_dir)
            shutil.copy(pic2_dir, cfg.test.g_dir)
            pic1_path = os.path.join(cfg.test.q_dir, pic1_name)
            pic2_path = os.path.join(cfg.test.p_dir, pic2_name)

            pic1 = cv2.imread(pic1_path)
            pic2 = cv2.imread(pic2_path)

            pic1 = cv2.cvtColor(pic1, cv2.COLOR_BGR2RGB)
            pic2 = cv2.cvtColor(pic2, cv2.COLOR_BGR2RGB)

            pic1 = cv2.resize(pic1, (256, 256))
            pic2 = cv2.resize(pic2, (256, 256))

            transf = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.ToTensor(mean = [0.5, 0.5, 0.5],
                                            std = [0.5, 0.5, 0.5])
                                        ])

            img1 = transf(pic1).cuda()
            img2 = transf(pic2).cuda()

            img1 = img1.view(1, *img1.size())
            img2 = img2.view(1, *img2.size())

            model.eval()
            with torch.no_grad():
                img1 = torch.cuda.FloatTensor(img1)
                img2 = torch.cuda.FloatTensor(img2)

                out1 = model(img1)
                out2 = model(img2)

            global_feat1 = out1["global_feat"]
            global_feat2 = out2["global_feat"]

            global_feat1 = F.normalize(global_feat1, dim=1, p=2)
            global_feat2 = F.normalize(global_feat2, dim=1, p=2)

            global_feat1 = global_feat1.detach().cpu()
            global_feat2 = global_feat2.detach().cpu()

            dismat = cosine_distance(global_feat1, global_feat2)
            print("dist:", dismat)
            if not os.path.exists("tsame.txt"):
                f = open("tsame.txt", "w")
            else:
                f = open("tsame.txt", "a")

            if pic1_name.split("_")[0] == pic2_name.split("_")[0]:
                issame = 1
            else:
                issame = 0

            test_str = str(issame) + " " + pic1_name + " " + pic2_name + " " + str(dismat[0][0]) + "\r\n"
            f.write(test_str)
            f.close()

            #计算一次后， 就将q与p删掉
            os.remove(pic1_path)
            os.remove(pic2_path)


        if __name__ == "__main__":
            main()