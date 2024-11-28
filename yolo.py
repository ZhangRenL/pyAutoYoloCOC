import os
import time
from typing import Dict, List, Tuple

import numpy as np
import random
import torch
import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
screensz = [0, 0, 1280, 720]
class YoloCOC:
    def __init__(self,
                 dec_model_path="./tools/runs/train1/yolo11COC640n/weights/best.pt",
                 cls_model_path="./tools/runs/classify/yolo11COC640n/weights/best.pt",
                 seg_model_path="./runs/segment/yolo11mseg-640/weights/best.pt"):
        # 加载模型
        self.dec_model = dec_model_path  # custom/local model
        self.cls_model = cls_model_path
        self.seg_model = seg_model_path
        self.img = None
        self.best_map = None

    def segment(self, image, imgsz=640,verbose=False, **kwargs):
        """

        :param image:
        :param imgsz:
        :return:
        返回包含两个元素的列表
        第一个元素：所有的xs
        第二个元素：所有的ys
        """
        h_ori,w_ori = image.shape[:2]
        if type(self.seg_model) is str:
            self.seg_model = YOLO(model=self.seg_model)
        if image is  None:
            return None
        re = self.seg_model(image, imgsz=imgsz, verbose=verbose, **kwargs)[0]
        if re.masks is None:
            return None
        masks = re.masks.data.cpu().numpy().astype(bool)
        if masks.shape[0] == 0:
            return None
        elif masks.shape[0] == 1:
            mask = masks[0]
        elif masks.shape[0] > 1:
            mask = masks[0]
            for i in range(1, masks.shape[0]):
                mask += masks[i]
        h_scal,w_scal = mask.shape
        loc = np.where(mask)
        return loc[1]/w_scal*w_ori, loc[0]/h_scal*h_ori



    def detect(self, image, imgsz=960, gray=True, show=False, all=False, wait_time=3, verbose=False,
               save=False, project=None, name=None):
        """识别所有非灰色元素
        Args:
            image (string/NumPy): image path or image
            range: 范围匹配
            size (int, optional): 图片尺寸. Defaults to 640.
            gray: 是否匹配灰色元素
        Returns:
            map: 返回confidence最大的一个元素
        """
        if type(self.dec_model)==str: self.dec_model = YOLO(self.dec_model)
        if image is None:
            return {}
        elif type(image) is str:
            image = cv2.imread(image)
        # 进行目标检测

        self.img=image
        result = self.dec_model(image, imgsz=imgsz, show=False, verbose=verbose, save=save,project=project, name=name)[0]
        if show:
            img = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
            self.dec_model(img, imgsz=imgsz, show=show, verbose=False)[0]
            cv2.waitKey(wait_time * 1000)
            cv2.destroyAllWindows()

        names = self.dec_model.names
        best_map = {}
        conf = {}
        all_map: dict[str, list[list[tuple[int, int]]]] = {}
        all_conf={}
        for box in result.boxes:
            name = names[int(box.cls)]
            confidence = float(box.conf)
            x1,y1,x2,y2 = box.xyxy.tolist()[0]

            # 获取范围
            x_range = (int(x1) * 1, int(x2) * 1)
            y_range = (int(y1) * 1, int(y2) * 1)
            if all:
                if name not in all_map.keys():
                    all_map[name] = [[x_range, y_range]]
                    all_conf[name] = [confidence]
                else:
                    all_map[name].append([x_range, y_range])
                    all_conf[name].append(confidence)
            else:
                if (name not in conf.keys() or confidence > conf[name]) and confidence > 0.3:
                    conf[name] = confidence
                    best_map[name] = [[x_range, y_range]]

        self.best_map = best_map
        self.all_map = all_map
        if all:
            return all_map
        else:
            return best_map

    def classify(self, image, imgsz=640,gray=True, show=False, all=False, wait_time=3,verbose=False, return_conf=False):
        if type(self.cls_model)==str: self.cls_model = YOLO(self.cls_model)
        if image is None:
            return {}
        elif type(image) is str:
            image = cv2.imread(image)
        # 进行目标检测
        # image = cv2.resize(image, (1600, 720))
        self.img=image
        result = self.cls_model(image, imgsz=imgsz, show=show, verbose=verbose)[0]
        probs = result.probs
        if show:
            img = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
            self.cls_model(img, imgsz=imgsz, show=show, verbose=False)[0]
            cv2.waitKey(wait_time * 1000)
            cv2.destroyAllWindows()
        names = self.cls_model.names
        re = []
        conf = []
        if all:
            for i in probs.top5: re.append(names[i])
            if return_conf:
                for i in probs.top5conf: conf.append(i.item())
        else:
            re.append(names[probs.top1])
            if return_conf: conf.append(probs.top1conf.item())
        if return_conf:
            return re, conf
        else:
            return re

    # TODO
    def stat_model(self):
        # worker_stats:
        # yanjiu_stats:
        # home_source_stats:
        # war_source_stats:
        # star_stats:
        # soldier_stats:
        # fashu_stats:
        # hero_stats:
        # gongcheng_stats:
        # 采集器——stats：
        # 仓库——stats：
        return None
    def show_best_map(self, wait=5):
        if self.img is not None and self.best_map is not None:
            img=cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            img=Image.fromarray(img)
            draw=ImageDraw.Draw(img)
            size=50
            font=ImageFont.truetype("C:/Windows/Fonts/SIMLI.TTF", size=size)
            for i in self.best_map.keys():
                color = (random.randint(0,125), random.randint(125,225), random.randint(0,255))
                x_range, y_range = self.best_map[i]
                draw.rectangle((x_range[0], y_range[0], x_range[1], y_range[1]), None,color, width=10)
                draw.text((x_range[0], y_range[0]-size if y_range[0]-size >= 0 else y_range[1] ), i, font=font, fill=(255,255,255))
            img=cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            cv2.imshow('img', cv2.resize(img, (1200,540)))
            cv2.waitKey(wait * 1000)
            cv2.destroyAllWindows()



def test_dec_model(show = False, wait=3, verbose=False, save=True):
    all = {}
    for model in ['yolo11coc960m', 'yolo11coc960s',
                  'yolo11coc1280m', 'yolo11coc1600m',
                  'yolo11coc1600s', 'yolo11coc1600n']:
        yolo = YoloCOC(dec_model_path='./tools/runs/train/' + model + '/weights/best.pt')
        # yolo.model.to("cpu")
        yolo.detect(None)
        names = yolo.dec_model.names
        for imgsz in [640, 960, 1280, 1600]:
            ka = model + "_" + str(imgsz)
            all[ka] = {}
            for n in names: all[ka][names[n]] = 0
            t0 = time.time()

            for img_n in os.listdir("data/test"):
                img = cv2.imread(os.path.join("data/test", img_n))
                img = cv2.resize(img, (1920, int(1080/2400*1920)))
                re = yolo.detect(img, show=show, wait_time=wait, imgsz=imgsz, all=True, verbose=verbose,
                                 save=save, project=os.path.join("test",ka), name=img_n)
                for k in re.keys(): all[ka][k] += len(re[k])
            t1=time.time()
            with open("YOLOtest.log", "a+") as f:
                lines=""
                print(ka, "time: %.3fs" % (t1-t0), end = "\t")
                lines += str(ka) + "\t" + "time: %.3fs" % (t1-t0) + "\t"

                for l in all[ka].keys(): print(l, all[ka][l], sep=": ", end="\t")
                for l in all[ka].keys(): lines += l + ": " + str(all[ka][l]) + "\t"
                print("", end = "\n")
                lines += "\n"
                f.write(lines)

    if save:
        for d in os.listdir("test"):
            for dd in os.listdir(os.path.join("test",d)):
                for ff in os.listdir(os.path.join("test",d,dd)):
                    os.rename(os.path.join("test",d,dd, ff),
                              os.path.join("test",d,dd.replace("png","")+ff))
                    os.rmdir(os.path.join("test",d,dd))
    # for k in all.keys():
    #     print(k, end="\t")
    #     for l in all[k].keys(): print(l, all[k][l], sep=": ",end="\t")
    #     print("", end="\n")

def test_cls_model(show = False, wait=3, verbose=True):
    all = {}

    for model in ['yolo11COC640n', 'yolo11COC640s',
                  'yolo11COC960n', 'yolo11COC960s']:
        yolo = YoloCOC(cls_model_path='./tools/runs/classify/' + model + '/weights/best.pt')
        # yolo.model.to("cpu")
        yolo.classify(None)
        names = yolo.cls_model.names
        for imgsz in [640, 960, 1280, 2400]:
            ka = model + "_" + str(imgsz)
            all[ka] = {}
            for n in names: all[ka][names[n]] = 0
            t0 = time.time()

            for img_n in os.listdir("data/test"):
                if not img_n.endswith(".jpg"):
                    continue
                img = cv2.imread(os.path.join("data/test", img_n))
                try:
                   img = cv2.resize(img, (1920, int(1080 / 2400 * 1920)))
                except:
                    print(img_n)
                    print(img.shape)
                re = yolo.classify(img, show=show, wait_time=wait, imgsz=imgsz, all=True, verbose=verbose)
                # print(re[0])
                all[ka][re[0]] += 1
            t1 = time.time()
            print(ka, "time: %.2fs" % (t1 - t0), end="\t")
            for l in all[ka].keys(): print(l, all[ka][l], sep=": ", end="\t")
            print("", end="\n")
if __name__ == '__main__':
    # test_dec_model(show=False, wait=2, verbose=False)
    test_dec_model(show=False, wait=2, verbose=False, save=True)
    test_dec_model(show=False, wait=2, verbose=False, save=False)

