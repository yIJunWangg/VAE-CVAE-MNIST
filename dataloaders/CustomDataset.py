import random
import torch
import torch.utils.data
from torchvision import transforms as TR
import os
from PIL import Image
from .utils import get_semantic_features
import numpy as np

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):
        opt.contain_dontcare_label = False
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0

        self.class_path = opt.class_dir
        with open(self.class_path, "r", encoding="utf-8") as f:
            cls = f.readlines()

        cls = cls[1:]
        self.class_dict = {}
        for c in cls:
            ls = c.split()
            self.class_dict[ls[0]] = ls[1]

        self.opt = opt
        self.for_metrics = for_metrics

        self.xpls, self.ppls, self.xpl_labels,self.ppl_labels, self.xpl_glcms, self.ppl_glcms, self.paths = self.list_images() # xpl,ppl, labels_xpl,labels_ppl, glcms_xpl,glcms_ppl, (path_img_xpl,path_img_ppl, path_lab)

    def __len__(self):
        return len(self.xpls)

    def __getitem__(self, idx):
        XPL = Image.open(self.xpls[idx]).convert('RGB')
        PPL = Image.open(self.ppls[idx]).convert('RGB')
        label_xpl = Image.open(self.xpl_labels[idx])
        label_ppl = Image.open(self.ppl_labels[idx])
        glcm_xpl = Image.open(self.xpl_glcms[idx])
        glcm_ppl = Image.open(self.ppl_glcms[idx])
        # 使用真实标签当作语义条件，计算类别比例
        sem_seg, sem_cond = get_semantic_features(torch.from_numpy(np.array(label_xpl)), self.opt.num_semantics, self.opt.label_unknown)
        (XPL,PPL), (label_xpl,label_ppl), (glcm_xpl,glcm_ppl) = self.transforms((XPL,PPL), (label_xpl,label_ppl), (glcm_xpl,glcm_ppl))    
        # label_xpl = label_xpl * 255
        # label_ppl = label_ppl * 255
        name_xpl = os.path.basename(self.xpls[idx])
        name_ppl = os.path.basename(self.ppls[idx])
        # cls = torch.tensor(int(self.class_dict[filename]))
        return {"xpl": XPL,"ppl":PPL, "label_xpl": label_xpl,"label_ppl":label_ppl, "name_xpl": name_xpl,"name_ppl":name_ppl, "glcm_xpl": glcm_xpl,"glcm_ppl":glcm_ppl,"sem_cond": sem_cond}#, 'class': cls}

    def list_images(self):
        mode = "test" if self.opt.phase == "test" or self.for_metrics else "train"
        # load image paths
        xpl = []
        path_img_xpl = os.path.join(self.opt.dataroot, mode + '_xpl')
        for city_folder in sorted(os.listdir(path_img_xpl)):
            item = os.path.join(path_img_xpl, city_folder)
            xpl.append(item)
        ppl = []
        path_img_ppl = os.path.join(self.opt.dataroot, mode + '_ppl')
        for city_folder in sorted(os.listdir(path_img_ppl)):
            item = os.path.join(path_img_ppl, city_folder)
            ppl.append(item)
        # load label paths
        labels_xpl = []
        path_lab = os.path.join(self.opt.dataroot, mode + '_label_xpl')
        for city_folder in sorted(os.listdir(path_lab)):
            item = os.path.join(path_lab, city_folder)
            labels_xpl.append(item)
        # labels_ppl = []
        # path_lab = os.path.join(self.opt.dataroot, mode + '_label_ppl')
        # for city_folder in sorted(os.listdir(path_lab)):
        #     item = os.path.join(path_lab, city_folder)
        #     labels_ppl.append(item)
        labels_ppl = []
        path_lab = os.path.join(self.opt.dataroot, mode + '_label_xpl')
        for city_folder in sorted(os.listdir(path_lab)):
            item = os.path.join(path_lab, city_folder)
            labels_ppl.append(item)
        
        # load glcm paths
        glcms_xpl = []
        path_canny = os.path.join(self.opt.dataroot, mode + '_glcm_xpl')
        for city_folder in sorted(os.listdir(path_canny)):
            item = os.path.join(path_canny, city_folder)
            glcms_xpl.append(item)
        glcms_ppl = []
        path_canny = os.path.join(self.opt.dataroot, mode + '_glcm_ppl')
        for city_folder in sorted(os.listdir(path_canny)):
            item = os.path.join(path_canny, city_folder)
            glcms_ppl.append(item)

        assert len(xpl)+len(ppl) == len(labels_xpl)+len(labels_ppl), "different len of images and labels %s - %s" % (len(xpl)+len(ppl), len(labels_xpl)+len(labels_ppl))

        return xpl,ppl, labels_xpl, labels_ppl, glcms_xpl,glcms_ppl, (path_img_xpl,path_img_ppl, path_lab)

    def transforms(self, images, labels, glcms):
        outimages=[]
        outlabels=[]
        outglcms=[]
        for i in range(len(images)):
            image= images[i]
            label = labels[i]
            glcm = glcms[i]
            # resize
            new_width, new_height = (int(self.opt.load_size / self.opt.aspect_ratio), self.opt.load_size)
            image = TR.functional.resize(image, (new_width, new_height), Image.BICUBIC)
            label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
            glcm = TR.functional.resize(glcm, (new_width, new_height), Image.NEAREST)
            assert image.size == label.size
            # flip
            if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
                if random.random() < 0.5:
                    image = TR.functional.hflip(image)
                    label = TR.functional.hflip(label)
                    glcm = TR.functional.hflip(glcm)
            # to tensor
            image = TR.functional.to_tensor(image)
            label = TR.functional.to_tensor(label)
            glcm = TR.functional.to_tensor(glcm)

            # normalize
            image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            outimages.append(image)
            outlabels.append(label)
            outglcms.append(glcm)
        return outimages, outlabels, outglcms
    