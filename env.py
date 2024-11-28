import random
import json
import sys
import atexit
import winsound
import os
from Others import *
import cv2
import time
from yolo import YoloCOC
import threading
# from env import *
# self = Env()
class Env:
    def __init__(self, device_id=None, cfg_path="env_cfg.json", wait_time_after_release_s=5,
                 reserve_worker=0, quite=False, debug=False,detail=0, screens=3):
        self.device_id = device_id
        self.screens = screens
        self.cfg = self.load_cfg(cfg_path)
        if os.path.exists(self.cfg["dec_model"]):
            dec_model_path = self.cfg["dec_model"]
        else:
            dec_model_path = "./tools/runs/train1/yolo11coc1280n/weights/best.pt"
        self.yolo = YoloCOC(dec_model_path=dec_model_path,
                            cls_model_path=self.cfg["cls_model"],
                            seg_model_path=self.cfg["seg_model"])

        self.detail = detail
        self.quite = True if detail == 0 else quite
        self.reserve_worker = reserve_worker

        self.log_click = False
        self.nm_ocr = cnocr.CnOcr(rec_model_name='densenet_lite_136-gru',
                                  det_model_name='naive_det',
                                  cand_alphabet='0123456789').ocr

        self.d = u2.connect(device_id)
        self.res = (max(self.d.window_size()), min(self.d.window_size()))
        t=time.time()
        self.time={"img": t,
                   "obj": t,
                   "cls": t,
                   "seg": t}
        self.img = self.d.screenshot(format="opencv")
        self.wait_time_after_release_s = wait_time_after_release_s
        self.last_strs=["ds"]
        self.war_time_s = 1e10

        # 九宫格密码每个点的坐标
        self.mima_jiu_gong_ge_pos = {1: (270, 1485), 2: (540, 1485), 3: (806, 1485),
                                     4: (270, 1750), 5: (540, 1750), 6: (806, 1750),
                                     7: (270, 2017), 8: (540, 2017), 9: (806, 2017)}
        # 阿拉伯数字密码每个数字的坐标
        self.mima_num_pos = {1: (270, 1485), 2: (540, 1485), 3: (806, 1485),
                             4: (270, 1750), 5: (540, 1750), 6: (806, 1750),
                             7: (270, 2017), 8: (540, 2017), 9: (806, 2017)}

        # 初始化后的自动程序
        randoms = np.random.normal(0, 1, 1000)
        self.randoms = (randoms / (max(randoms) - min(randoms))).tolist()
        self.ocr_f = cnocr.CnOcr().ocr
        self.d.stop_uiautomator()
        self.d.start_uiautomator()
        if not debug:
            atexit.register(self.exit)
        # 后台进程
        # TODO
        self.stop_flag = True
        self.thread = threading.Thread(target=self.find_unknown_page)
        self.thread.daemon = True
        self.thread.start()
    def img_time(self,update=None):
        if update is None: return self.time["img"]
        else: self.time["img"] = time.time()

    def obj_time(self,update=None):
        if update is None: return self.time["obj"]
        else: self.time["obj"] = time.time()

    def cls_time(self,update=None):
        if update is None: return self.time["cls"]
        else: self.time["cls"] = time.time()

    def seg_time(self,update=None):
        if update is None: return self.time["seg"]
        else: self.time["seg"] = time.time()

    def load_cfg(self, path):
        with open(path, 'r') as f:
            cfg = json.load(f)
            return cfg

    def stop_find_unknown_page(self):
        self.stop_flag=False

    def find_unknown_page(self):
        img_time = self.img_time()
        while self.stop_flag:
            time.sleep(random.randint(30, 60))
            if self.img is not None:
                if img_time == self.img_time():
                    continue
                img=self.img
                img_time = self.img_time()
                re,conf = self.yolo.classify(img, show=False, return_conf=True)
                re, conf = re[0], conf[0]
                if conf <= 0.95:
                    if not os.path.exists("data/unknown_pages"): os.mkdir("data/unknown_pages")
                    filename = "data/unknown_pages/%d.png" % time.time()
                    cv2.imwrite(filename, img)
                    with open("data/unknown_pages/CLSunknown.txt", "a+") as f:
                        f.write(filename + ":\t" +re + "\t" + str(conf) + "\n" )
                        if not self.quite:
                            print("recored")



    def ocr(self, img, x_range=None, y_range=None):
        if x_range is not None:
            img[:, :min(x_range)] = 0
            img[:, max(x_range):] = 0
        if y_range is not None:
            img[:min(y_range), :] = 0
            img[max(y_range):, :] = 0
        res = self.ocr_f(img)
        self.last_ocr_res = res
        return res


    def classify(self, show=False, wait=3, img=None):
        if img is None:
            self.update_img()
            img=self.img
        re = self.yolo.classify(img, show=show, wait_time=wait)
        self.cur_page = re[0]
        self.cls_time(1)
        return re

    def update_cur_page(self, img=None):
        self.classify(img=img)


    def update_objs(self, show=False, wait=3, img=None, imgsz=None, all=False,
                    x_range=None, y_range=None, **kwargs):
        if img is None or time.time() - self.img_time() > 0.5:
            self.update_img()
            img=self.img.copy()
        if x_range is not None:
            img[:, :min(x_range)]=0
            img[:, max(x_range):]=0
        if y_range is not None:
            img[:min(y_range), :] = 0
            img[max(y_range):, :] = 0

        self.obj_time(1)
        if imgsz is None: imgsz=round(max(self.res)/2/32)*32
        self.objs = self.yolo.detect(img,
                                     show=show, wait_time=wait, imgsz=imgsz,
                                     all=all, **kwargs)

    def click_obj(self, obj, objs=None, t=0.1):
        if objs is None:
            self.update_objs()

        if obj in self.objs.keys():
            x_range, y_range = self.objs[obj][0]
            self.random_click(x_range, y_range, t=t)
            return True
        else:
            return False

    def collect_rewards(self, show=False, wait=3):
        self.update_objs()
        if "奖励_y" in self.objs.keys():
            self.click_obj("奖励_y", self.objs)
            self.random_wait(0.5)
            self.click_txt("奖励")
            time.sleep(wait)
            flag=True
            while flag:
                flag = False
                self.update_objs()
                for i in ["领取", "更多"]:
                    if i in self.objs.keys():
                        flag = True
                        self.click_obj(i)
                        self.random_wait(0.5)
                        self.update_objs()
                self.random_wait(0.5)

        if not self.click_obj("关闭"): self.d.keyevent('back')

        self.random_wait(wait)
        self.update_objs()
        if "钻石奖励" in self.objs.keys():
            self.click_obj("钻石奖励")
            time.sleep(2)
            if not self.click_obj("领取"):
                img = self.img.copy()
                if not self.click_txt("领取奖励", img=img): self.click_txt("领取", img=img)
            time.sleep(1)
            if not self.click_obj("关闭"): self.d.keyevent('back')

    def reset(self):
        if self.d.info['screenOn'] is False:
            return False

        n=0
        while True:
            n+=1
            if n > 10:
                winsound.Beep(1000, 200)
                winsound.Beep(1000, 200)
                break
            time.sleep(0.5)
            self.update_img(True)
            self.update_cur_page()
            winsound.Beep(1000, 200)
            if self.cur_page in ["home_zsj", "home_ysj"]:
                return True
            elif self.cur_page == "wait":
                pass
            elif self.cur_page.startswith("war") or self.cur_page == "counting":
                self.d.keyevent('back')
                self.random_wait(1)
                self.click_obj("确定")
                self.random_wait(1)
                # self.click_obj("回营")
                # self.random_wait(1)
            elif self.cur_page == "star_reward":
                self.d.keyevent('back')
            elif not self.cur_page in ["home_zsj", "home_ysj"]:
                self.d.keyevent('back')
            else:
                winsound.Beep(2000, 200)
                winsound.Beep(3000, 200)
                break
            time.sleep(0.5)


    def update_stats(self, showstrs=False):
        self.stats = {'进攻': -1, '防守': -1, '退出':-1,
                      '立即寻找': -1,"搜索对手":-1,"单人模式":-1, "准备就绪":-1,
                      '取消': -1, '开战': -1, '战倒计时': -1,
                      '放弃': -1, '回营': -1, '结束战': -1,
                      '商店': -1, '还有': -1, '确定': -1,
                      '重试': -1, '重新': -1, '摧毁率%': -1,
                      '已派出所有兵力': -1, "请选择其他兵种": -1,
                      '红线':-1,"需要部队":-1, "下一": -1, '部落消息': -1,
                      '消息': -1, "需要更多的": -1, "缺少": -1
                      }
        time.sleep(0.2)
        txts = self.get_strs()
        txts.extend(self.get_strs(self.img[:,800:1600,:]))
        self.last_strs=txts
        for s in txts:
            for k in self.stats.keys():
                if s.find(k) > -1:
                    self.stats[k] = s.find(k)
            if s.find("%") > -1:
                self.stats['摧毁率%'] = s.replace("%", "").replace("o", "0").replace("O", "0").replace("D", "0")
                try:
                    self.stats['摧毁率%'] = int(self.stats['摧毁率%'])
                except:
                    self.stats['摧毁率%'] = -1
                    # sys.exit()
            if s.find('经派出所有兵力') > -1 or s.find('经派出') > -1 or s.find('经派') > -1 or s.find('所有兵') > -1:
                self.stats['已派出所有兵力'] = 1
            if s.find('请选择其他兵种') > -1 or s.find('请选择') > -1 or s.find('他兵种') > -1 or s.find('其他兵') > -1:
                self.stats['请选择其他兵种'] = 1
            if s.find("些部队") > -1 or s.find("需要") > -1:
                self.stats["需要部队"] = 1
            # if s.find('正在载入') > -1:
            #     self.random_wait(30)

        if self.stats['重试'] > -1:
            self.click_txt('重试')
            self.random_wait(4)
            self.update_stats()
        if self.stats['重新'] > -1:
            self.click_txt('重新')
            self.random_wait(4)
            self.update_stats()
        if not self.quite:
            print(self.stats)

    def update_img(self, force=False):
        if force:
            update = True
        elif time.time() - self.img_time() > 0.2:
            update = True
        else:
            update = False

        if update:
            self.img = self.d.screenshot(format="opencv")
            self.img_time(1)
            filename = "data/" + str(round(time.time())) + ".png"
            # if True:
            #     cv2.imwrite(filename=filename, img = self.img)
            if not self.quite:
                print("update img time: ", time.localtime().tm_hour, ": ", time.localtime().tm_min, ": ",
                      time.localtime().tm_sec)
            else:
                pass

    def findTxt(self, string):
        return self.click_txt(string, click=False)


    def get_strs(self, img=None, x_range=None, y_range=None):
        if x_range is not None or y_range is not None:
            self.update_img()
            img = self.img.copy()
            if x_range is not None:
                img[:, :min(x_range)] = 0
                img[:, max(x_range):] = 0
            if y_range is not None:
                img[:min(y_range), :] = 0
                img[max(y_range):,:] = 0

        if img is None:
            return self.click_txt("NIUBIPLUS", click=False, return_str=True)
        else:
            return self.click_txt("NIUBIPLUS", click=False, return_str=True, img=img)

    def click_txt(self, string, click=True, return_str=False, img=None):
        """

        : param string: 要查找的字符串
        : param click: 是否点击该字符串
        : return: 如果找不到则返回None，如果找到则返回该字符串的位置
        """
        if not (self.quite) and string != "NIUBIPLUS":
            print("Click " + str(string) + " Finding")
        if type(string) == type("dfghjk"):
            string = [string]
        if img is not None:
            res = self.ocr(img)
        else:
            self.update_img()
            res = self.ocr(self.img)
        x_range, y_range = None, None
        txts = []
        for item in res:
            if item['score'] < 0.35: continue
            txts.append(item['text'])
            for s in string:
                if item['text'].find(s) > -1:
                    pos = item['position']
                    x1, y1, x2, y2 = pos[0, 0], pos[0, 1], pos[2, 0], pos[2, 1]
                    x_range = (x1, x2)
                    y_range = (y1, y2)
                if click:
                    break
        if return_str:
            return txts
        if x_range is not None:
            if not self.quite:
                print("Click ", string, " Found")
            if click:
                self.random_click(x_range, y_range)
                return x_range, y_range
            else:
                return x_range, y_range
        else:
            if not self.quite:
                print("Click ", string, "Not Found")
            return None

    def get_txt_pos(self, txt):
        return self.click_txt(txt, click=False)


    def clickpos(self, pos, t=0.1, quite=True):
        self.d.long_click(pos[0], pos[1], t)
        if not (self.quite or quite):
            print('点击了: ' + str(pos))
        if self.log_click:
            with open(self.device_id[4:] + "clicklog.txt", "a") as f:
                f.write(time.strftime("%Y-%m-%d %H: %M: %S", time.localtime()) + '\t%d\t%d\n' % (pos[0], pos[1]))

        self.random_wait(0.2)
        if self.d.device_info["arch"] == "x86_64":
            time.sleep(1)
        else:
            time.sleep(0.1)

    def random_click(self, x_range, y_range, t=0.1, quite=True):
        s = random.sample(self.randoms, 2)
        x = s[0] * (max(x_range) - min(x_range)) + (max(x_range) + min(x_range)) / 2
        y = s[1] * (max(y_range) - min(y_range)) + (max(y_range) + min(y_range)) / 2

        quite = (self.quite or quite)
        if t > 1:
            t = t + random.random() * t
        self.clickpos((int(x), int(y)), quite=quite, t=t)

    def random_wait(self, t=2.0, quite=True, m=""):
        t = (0.5 - random.random()) * t + t
        if not quite:
            print("wait time:", round(t, 2), end = " s" + str(m))

        time.sleep(t-int(t))
        t=int(t)
        if not quite:
            print("\rwait time:", t, "s", m,end="")
        time.sleep(t%20)
        t = t - t % 20
        while t > 0:
            if t > 20:
                if not quite:
                    print("\r ", end="")
                    print("\rwait time:", t, "s", m, end="")
                time.sleep(20)
                t = t - 20
            else:
                if not quite:
                    print("\r ", end="")
                    print("\rwait time:", t, "s", m, end="")
                time.sleep(1)
                t = t - 1
        if not quite:
            print("\r ", end = "")


    def click_pic(self, pic_target, MIN_NUM_GOOD_MATCHES=3, t=0.1, quite=True, click=True, update_img=True):
        if not os.path.exists(pic_target):
            print(pic_target, "not exists")
        # img_screen_shot = self.d.screenshot(format="opencv")
        # pts = matchpic(pic_target, img_screen_shot,
        if update_img:
            self.update_img()
        pts = matchpic(pic_target, self.img,
                       MIN_NUM_GOOD_MATCHES=MIN_NUM_GOOD_MATCHES,
                       show=not quite)
        if not pts is None and len(pts) > 0:
            # # 随机获取其中一个匹配到的特征点
            # pt = random.choice(pts)
            # self.clickpos(pt, quite)
            # return pt

            # 随机返回特征点范围内的一个点
            xs = []
            ys = []

            for p in pts:
                xs.append(p[0])
                ys.append(p[1])
            x_range = (min(xs), max(xs)) if not min(xs) == max(xs) else min(xs)
            y_range = (min(ys), max(ys)) if not min(ys) == max(ys) else min(ys)
            if click:
                self.random_click(x_range, y_range, quite=quite, t=t)
            time.sleep(0.5)
            if not quite:
                try:
                    cv2.imshow('img', self.img[x_range[0]:x_range[1], y_range[0]:y_range[1],:])
                    cv2.waitKey(3000)
                    cv2.destroyAllWindows()
                except:
                    pass
            return x_range, y_range

        else:
            if not (
                    self.quite):
                print("target not found")
            return None

    def goto_zsj(self):
        self.reset()
        self.update_cur_page()
        if self.cur_page == "home_zsj":
            return True

        self.d().pinch_in(50,50)
        w, h = (max(self.res), min(self.res))
        for _ in range(10):
            fx,fy,tx,ty=w*1000/2400,h*700/1080,w*300/2400,h*800/1080
            self.d.swipe(int(fx), int(fy), int(tx), int(ty), 0.5)
        self.d.swipe(int(w*1000/2400), int(h*700/1080), int(w*300/2400), int(h*800/1080), 0.5)
        # self.d().pinch_out(6, 6)
        self.d().swipe("left", 20)
        self.random_wait(0.4)
        self.click_obj('bout')
        time.sleep(3)

        self.reset()
        self.update_cur_page()
        if self.cur_page == "home_zsj":
            return True
        else:
            return False


    def goto_ysj(self):
        self.reset()
        self.update_cur_page()
        if self.cur_page == "home_ysj":
            return True
        self.d().pinch_in(50, 50)
        w,h=max(self.res),min(self.res)
        fx,fy,tx,ty=w*400/2400,h*800/1080,w*400/2400,h*400/1080
        self.d.swipe(int(fx),int(fy),int(tx),int(ty), 0.05)
        self.random_wait(0.3)
        self.d.swipe(int(fx),int(fy),int(tx),int(ty), 0.05)
        # self.d().pinch_out(6, 6)
        self.d().swipe("right", 4)
        self.random_wait(0.4)
        self.click_obj('bout')
        time.sleep(3)

        self.reset()
        self.update_cur_page()
        if self.cur_page == "home_ysj":
            return True
        else:
            return False


    def get_war_resources(self):
        self.update_img()
        x,y = int(max(self.res)/3), int(min(self.res)/3)
        img = self.img.copy()
        strs = self.get_strs(img, (0,x), (0, y))

        # strs = self.get_strs(self.img[:x,:y, :])
        gold, water, oil = -1, -1, -1
        for n in range(strs.__len__()):
            if strs[n].find("可获得的战利品") > -1 or strs[n].find("可获得") > -1 or strs[n].find("战利品") > -1:
                num=[]
                for s in strs[n+1:]:
                    s = s.replace(" ", "").replace("O", "0").replace("o", "0").replace("B", "8").replace("-","")
                    s = s.replace("S", "5").replace("s", "5").replace("d", "0").replace("D", "0")
                    try:
                        s = int(s)
                    except ValueError:
                        s = -1
                    if s > 100:
                        num.append(s)
                if num.__len__() == 3:
                    gold, water, oil =num
                elif num.__len__() == 2:
                    gold, water = num
                else:
                    pass
                    # print("fuck........\t", strs)
                    # print("gold: %d, water : %d, oil: %d" % (gold, water, oil))

        # cv2.imshow("war", self.img[:x,:y])
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()
        return int(gold), int(water), int(oil)


    def shou_ji_sheng_shui(self):
        time.sleep(1)
        self.update_stats()
        if self.stats['确定'] > -1:
            self.click_txt('确定')
        print("收集", end="")
        self.update_stats()
        if not self.quite:
            print(self.stats)

        self.d().pinch_in(50, 20)
        w,h=max(self.res),min(self.res)
        fx,fy,tx,ty=w*400/2400,h*300/1080,w*300/2400,h*900/1080
        self.d.swipe(int(fx),int(fy),int(tx),int(ty), duration=0.05)
        for _ in range(5):
            fx, fy, tx, ty = w * 1000 / 2400, h * 700 / 1080, w * 300 / 2400, h * 800 / 1080
            self.d.swipe(int(fx),int(fy),int(tx),int(ty), duration=0.05)
            self.random_wait(0.1)
        self.d().pinch_out(8, 10)

        self.d.swipe(w/2, h/2, w/2-500, h/2-100, 0.1)
        self.random_wait(1)

        print("\r查找圣水车", end="")
        if self.click_obj("收集"):
            # TODO 删除1行
            self.d.screenshot("data/COC_cls/train/collect/" + str(round(time.time())) + ".png")
            self.random_wait(0.5)
            if not self.click_txt("收集"): self.click_obj("收集")
            self.random_wait(0.5)
            self.click_obj("关闭")
        self.random_wait(0.5)



    def zhao_yu(self, condition, retry=True):
        """
        condition:
        water > 40000 and gold > 40000 and oil > 5000
        water + gold > 1000000
        oil > 5000
        gold > 800000
        water > 100000
        : param condition:
        : return:
        True: finished
        -1: 需要部队
        """
        self.update_cur_page()
        if self.cur_page != "home_zsj":
            self.goto_zsj()
        n=0
        while True:
            n+=1
            self.update_objs()
            objs = self.objs
            if self.click_obj("进攻_胜利之星", objs): break
            if self.click_obj("进攻", objs): break
            time.sleep(1.5)
            if n > 10: break

        if not self.click_obj("立即寻找"): self.click_txt("搜索")
        self.update_objs()
        if "需要部队" in self.objs.keys(): return -1
        self.update_stats()
        if self.stats["需要部队"] > -1: return -1
        time.sleep(1.5)

        tri_n = 0
        m = condition
        while True:
            tri_n += 1
            if tri_n > 50:
                self.zhao_yu(condition=condition, retry=False)
            t = 0
            self.random_wait(2, m=m)
            self.update_cur_page()
            if self.cur_page=="home_zsj":
                if retry: self.zhao_yu(condition=condition,retry=False)
                else: break
            elif self.cur_page=="warning": break
            elif self.cur_page=="others": time.sleep(3)

            while (self.cur_page not in ["counting"]):
                time.sleep(2)
                self.update_cur_page()
                t += 1
                if t > 5: break
            gold, water, oil = self.get_war_resources()
            m = "Need:" + condition + "Cur: water=%d,gold=%d,oil=%d" % (water, gold, oil)
            if not self.quite:
                print("gold: %d, water: %d, oil: %d" % (gold, water, oil))

            if eval(condition):
                winsound.MessageBeep(0)
                break

            # 点击下一个
            self.click_obj("下一个")

        if not self.quite:
            print("成功找到大户")
        return True


    def unlock(self, pwd=(3, 2, 4, 7, 8, 6, 9), pwd_type="jgg"):
        self.d.screen_off()
        self.d.unlock()
        time.sleep(1)

        # self.d.swipe_points((self.mima_jiu_gong_ge_pos[8], self.mima_jiu_gong_ge_pos[2]), 0.05)
        time.sleep(1)
        if pwd_type in ["jgg", "JGG", "jiugongge", "JiuGongGe"]:
            """
            {1: (270,1485),2: (540,1485),3: (806,1485),
              4: (270,1750),5: (540,1750),6: (806,1750),
              7: (270,2017),8: (540,2017),9: (806,2017)}
              """
            points = []
            for i in pwd:
                points.append(self.mima_jiu_gong_ge_pos[i])
            self.d.swipe_points(points, 0.05)
            time.sleep(1)
            # self.d.keyevent('home')


    def lanch_app_by_name(self, txt="部落冲突", screens=None, retry=True):
        """

        : param icon: app的icon,png格式
        : param screens: app在第几屏，+数据为home页右边；-数为在home页左边
        : return:
        """
        if screens is None:
            screens=self.screens
        time.sleep(1.5)
        self.d.screen_off()
        time.sleep(0.5)
        self.unlock()

        self.d.keyevent('home')
        time.sleep(1)
        self.d.keyevent('home')
        time.sleep(1)
        for _ in range(abs(screens)):
            if screens > 0:
                self.d().swipe("left", 10)
            else:
                self.d().swipe("right", 10)
            time.sleep(1)

        r = self.click_txt(txt)
        if r is not None:
            print("打开", txt, "成功")
            self.random_wait(5)
            self.reset()
        else:
            print("打开失败")
            if retry:
                self.lanch_app_by_name(txt, screens, retry=False)
            else:
                sys.exit()

    def switch_users(self, username):
        suc=False
        self.reset()
        time.sleep(1)
        self.click_obj("设置")
        self.random_wait(1)
        r=self.click_obj("切换")
        self.random_wait(1)
        if r is None:
            return True

        n = 0
        while True:
            r = self.click_txt(username)

            if r is None:
                w = int(max(self.res)/3*2)
                h = int(min(self.res))
                self.d.swipe(0.8*w,0.8*h,0.7*w,0.5*h, 0.25)
                time.sleep(0.5)
                n += 1
                if n > 10:
                    break
            else:
                suc=True
                break
        time.sleep(10)
        self.update_stats()
        return suc

    def exit(self):
        """
        回到home页面，并关闭屏幕
        : return:
        """
        self.d.keyevent('home')
        time.sleep(1)
        self.d.keyevent('home')
        time.sleep(1)
        self.d.screen_off()
        winsound.Beep(9000, 600)
        winsound.Beep(9000, 450)
        winsound.Beep(9000, 300)

    def update_source_stats(self):
        self.source_stats = {"gold": -1,
                             "water": -1,
                             "oil": -1,
                             "zuan": -1}
        self.update_img(force=True)
        strs = self.get_strs(img=self.img[: 400, -500:, :])
        nums = []
        for STR in strs:
            num = STR
            num = num.replace(" ", "").replace("O", "0").replace("o", "0").replace("B", "8")
            num = num.replace("S", "5").replace("s", "5").replace("d", "0").replace("D", "0")
            try:
                num = int(num)
                nums.append(num)
            except:
                pass
        if len(nums) == 4:
            self.source_stats["gold"] = nums[0]
            self.source_stats["water"] = nums[1]
            self.source_stats["oil"] = nums[2]
            self.source_stats["zuan"] = nums[3]
        elif len(nums) == 3:
            self.source_stats["gold"] = nums[0]
            self.source_stats["water"] = nums[1]
            self.source_stats["zuan"] = nums[2]
        else:
            pass
            # print("fucks")
            # print(strs)

    def auto_train_zsj(self):
        tries=0
        while self.findTxt("军队") is None:
            self.click_obj("训练")
            self.random_wait(1)
            tries+=1
            if tries >=10:
                break

        self.click_txt("键训练")
        try:
            for _ in range(2):
                self.update_img(force=True)
                _, y1 = self.get_txt_pos("AAA")
                _, y2 = self.get_txt_pos("BBB")

                self.img[:int(max(y1))] = 0
                self.img[int(min(y2)):] = 0

                # cv2.imshow("img", self.img)
                # cv2.waitKey(3000)
                # cv2.destroyAllWindows()
                self.click_txt("训练", img=self.img)
                self.update_stats()
                if self.stats['确定'] > -1:
                    self.click_txt("确定")
                time.sleep(0.5)
        except Exception as e:
            print("Wrong", end="")
            def train():
                while True:
                    if random.random() < 0.5: self.d.swipe(0.6, 0.6, 0.4, 0.3, steps=30)
                    else: self.d.swipe(0.4, 0.6, 0.7, 0.4, steps=30)

                    self.update_objs(all=True, y_range=(int(min(self.res)/2), max(self.res)))
                    avs = []
                    for i in self.objs.keys():
                        if i.endswith("9"):
                            avs.append(i)
                    if avs.__len__() > 0:
                        i = random.choice(avs)
                        self.click_obj(i, self.objs, t=random.randint(10,200)/100)
                        time.sleep(1)
                    else:
                        break

            if self.click_txt("训练部队") is not None:
                train()
            if self.click_txt("配制法术") is not None:
                train()
            if self.click_txt("攻城机器") is not None:
                train()

        self.d.keyevent("back")




    def show_img(self, img = None, resize=0.5, x_range=None, y_range=None):
        if img is None:
            self.update_img()
            img = self.img.copy()
        else:
            pass
        if x_range is not None:
            img[:, :min(x_range)] = 0
            img[:, max(x_range):] = 0
        if y_range is not None:
            img[:min(y_range), :] = 0
            img[max(y_range):, :] = 0
        img = cv2.resize(img, None, fx=resize, fy=resize)
        cv2.imshow("img", img)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

    # def get_str_pos(self, img=None, x_range=None, y_range=None):
    #     """
    #
    #     :param img:
    #     :return:
    #     两个列表
    #     列表1：所有文字
    #     列表2：与文字对应的位置（字典）
    #            {
    #            "xfrom": int(min(x_range)),
    #            "xto": int(max(x_range)),
    #            "yfrom": int(min(y_range)),
    #            "yto": int(max(y_range))
    #            }
    #     """
    #
    #     if x_range is None or y_range is None:
    #         self.update_img()
    #         img = self.img.copy()
    #         if x_range is not None:
    #             img[:, :min(x_range)] = 0
    #             img[:, max(x_range):] = 0
    #         if y_range is not None:
    #             img[:min(y_range), :] = 0
    #             img[max(y_range):, :] = 0
    #
    #     if img is None:
    #         self.update_img()
    #         img = self.img.copy()
    #     strs = self.ocr(img)
    #     res = []
    #     pos = []
    #     for i in strs:
    #         res.append(i["text"])
    #         posi = i['position']
    #         x1, y1, x2, y2 = posi[0, 0], posi[0, 1], posi[2, 0], posi[2, 1]
    #         x_range = (x1, x2)
    #         y_range = (y1, y2)
    #         pos.append({"xfrom": int(min(x_range)), "xto": int(max(x_range)),
    #                     "yfrom": int(min(y_range)), "yto": int(max(y_range))})
    #     return res,pos


    def auto_fight(self,
                   zsj_first=True,
                   return_zsj=True,
                   condition="water > 300000 and gold > 300000",
                   ysj_jg_time=9,
                   ysj_sj_time=1,
                   detail=0):
        """
        condition={'老机吧大':"water > 800000 and gold > 800000 and oil > 8000",
                    '随法':"water > 300000 and gold > 300000",
                    '泊松':"water > 300000 and gold > 300000",
                    '私域':"water > 300000 and gold > 300000",
                    '源代码':"water > 800000 and gold > 800000 and oil > 5000"}
        :param users:
        :param conditions:
        :param detail:
        :param epochs:
        :param inters:
        :return:
        True: finished
        False: failed
        """

        def fight_zsj(condition):
            print("fight_z", end="    ")
            if self.cur_page == "home_ysj": self.goto_zsj()
            for _ in range(2):
                self.update_source_stats()
                if self.source_stats["gold"] % 500 == 0 and self.source_stats["water"] % 500 == 0:
                    print("storage is full, exit...")
                    break
                #
                if condition is None or condition is True:
                    condition = 'True'
                r = self.fight_yolo(condition=condition)
                self.random_wait(3)
                if r == -1 or "需要部队" in self.objs.keys():
                    print("train_z", end="    ")
                    self.auto_train_zsj()
                    return False
                self.update_cur_page()
                if self.cur_page == "star_reward": self.click_obj("确定")
            print("train_z", end="    ")
            self.auto_train_zsj()

        def fight_ysj(ysj_jg_time,ysj_sj_time):
            # Fight in YSJ
            print("goto_y", end="    ")
            if self.cur_page == "home_zsj": self.goto_ysj()
            self.reset()
            self.update_cur_page()
            if self.cur_page != "home_ysj": return None
            wait_time = self.wait_time_after_release_s
            print("fight_y", end="    ")
            for sj_time in range(ysj_sj_time):
                print("sj:", sj_time, end=".")
                self.random_wait(3)
                ysj_jg_time = random.randrange(max(1, ysj_jg_time - 2), max(3, ysj_jg_time + 2))
                for jg_time in range(ysj_jg_time):
                    print(jg_time + 1, end=".")
                    print('\rsj_time: %d/%d; inters: %d/%d' % (sj_time + 1, ysj_sj_time, jg_time + 1, ysj_jg_time),
                          end="")
                    if (ysj_jg_time - jg_time == 1 or random.random() < 0.1) and self.wait_time_after_release_s < 20:
                        self.wait_time_after_release_s = 120
                        print("\rwait for release")
                    else:
                        self.wait_time_after_release_s = wait_time
                        time.sleep(3)
                    self.update_source_stats()
                    if self.source_stats['water'] % 10000 == 0 and self.source_stats['gold'] % 10000 == 0:
                        print("\rStorage is full, exit...")
                        break

                    if self.source_stats['gold'] % 10000 == 0 and self.source_stats['gold'] > 0:
                        # 金满
                        self.wait_time_after_release_s = random.choice([2, 3, 4, 5, 6])
                    elif self.source_stats['water'] % 10000 == 0 and self.source_stats['water'] > 0:
                        # 水满
                        self.wait_time_after_release_s = 120
                    else:
                        self.wait_time_after_release_s = wait_time

                    # 正主在这
                    self.fight_yolo()
                self.shou_ji_sheng_shui()

            self.wait_time_after_release_s = wait_time

        def upgrade_clear():
            print("rm_zw", end="    ")
            self.remove_zw()
            print("collect", end="    ")
            self.collectandacclerate()
            print("upgrade", end="    ")
            for _ in range(3 if world=="home_zsj" else 2):
                self.upgrade()
            print("rewards", end="    ")
            self.collect_rewards()
            print("\r ")

        # 开始处理
        self.reset()
        time.sleep(1)
        self.update_cur_page()
        if self.cur_page in ["home_zsj", "home_ysj"]:
            world = self.cur_page
        else:
            print("reset fail")
            return False

        self.log_click = True

        if zsj_first :
            if world == "home_ysj":
                self.goto_zsj()
            fight_zsj(condition=condition)
            upgrade_clear()
            time.sleep(5)
            fight_ysj(ysj_jg_time=ysj_jg_time,ysj_sj_time=ysj_sj_time)
            upgrade_clear()
        else:
            if world == "home_zsj":
                self.goto_ysj()
            fight_ysj(ysj_jg_time=ysj_jg_time,ysj_sj_time=ysj_sj_time)
            upgrade_clear()
            time.sleep(5)
            fight_zsj(condition=condition)
            upgrade_clear()

        self.log_click = False

        self.reset()
        self.update_cur_page()
        if return_zsj and self.cur_page == "home_ysj":
            self.goto_zsj()
            self.upgrade()
        return None


    def auto_fight_multi_users(self, users, conditions, ysj_sj_times=None, ysj_jg_times = None):
        all_user = users.copy()
        self.lanch_app_by_name()
        n = 0
        last_n = 0
        for _ in range(20):
            users = random.sample(all_user, random.choice([users.__len__() - 1, users.__len__()]))
            for user in users:
                n += 1
                print(n, user[0])
                rn = random.random()
                if (rn > 0.9 or last_n - n >= 15) and n > 2:
                    last_n = n
                    if n > 200:
                        print(n)
                        return n
                    self.exit()
                    for i in range(0, 300, 10):
                        print("\r rn:", rn, " sleep:", 300 - i, end="s")
                        time.sleep(10)
                    self.lanch_app_by_name()
                r = self.switch_users(user)
                if not r:
                    continue
                try: condition = conditions[user]
                except Exception as e: condition = "True"
                try: ysj_sj_time = ysj_sj_times[user]
                except Exception as e: ysj_sj_time = random.choice([1,2, 1, 1])
                try: ysj_jg_time = ysj_jg_times[user]
                except Exception as e: ysj_jg_time =  random.randint(4,8)
                self.auto_fight(condition=condition,
                                ysj_sj_time=ysj_sj_time,
                                ysj_jg_time=ysj_jg_time)
    def is_upgrader_available(self, upgrader='工人'):
        """
        upgrader='工人'
        upgrader='研究'
        :param upgrader:
        :return:
        """
        availible = False
        self.update_objs()
        if upgrader not in self.objs.keys():
            availible = False
        else:
            (x1, x2), (y1, y2) = self.objs[upgrader][0]
            w = int((x2 - x1) * 1.5)
            x2 += w
            img=self.img.copy()
            # TODO 删除
            img=img[y1:y2, x1:x2, :]
            cv2.imwrite("data/worker_cls/" + str(round(time.time())) + ".png", img)
            # cv2.imshow('img',img)
            # cv2.waitKey(3000)
            # cv2.destroyAllWindows()
            # str_c = self.ocr(img, x_range=(x1, x2), y_range=(y1, y2))
            str_c = self.ocr(img)
            str_c = str_c[0]['text']
            str_c = str_c.replace(" ", "").replace("O", "0").replace("o", "0").replace("B", "8").replace("A", "1")
            str_c = str_c.replace("S", "5").replace("s", "5").replace("d", "0").replace("D", "0")
            try:
                if int(str_c[0]) != 0:
                    availible = True
            except:
                pass
        return availible

    def upgrade(self):
        # for i in ['工人']:
        for i in ['工人', '研究']:
            while True:
                self.update_objs()
                if i in self.objs.keys():
                    pass
                else:
                    continue

                (x1, x2), (y1, y2) = self.objs[i][0]
                w = int((x2 - x1) * 1.5)
                x2 += w
                img=self.img.copy()
                # TODO 删除
                img=img[y1:y2, x1:x2, :]
                cv2.imwrite("data/worker_cls/" + str(round(time.time())) + ".png", img)
                str_c = self.ocr(img)
                str_c = str_c[0]['text']
                str_c = str_c.replace(" ", "").replace("O", "0").replace("o", "0").replace("B", "8").replace("A","1").replace(",","")
                str_c = str_c.replace("S", "5").replace("s", "5").replace("d", "0").replace("D", "0").replace("V","1/").replace(".","")

                try:
                    int(str_c[0])
                except:
                    print("Wrong with int(str):", str_c)
                    str_c = "0"
                if int(str_c[0]) == 0:
                    # 工人不可用
                    break
                if int(str_c[0]) == self.reserve_worker and i == '工人':
                    # 保留工人刷墙
                    break

                # 工人可用时
                if not self.quite: print("worker is aviliable.")
                self.random_click((x1, x2), (y1, y2))
                self.update_objs()
                # 有可升级项目
                if '可升级' not in self.objs.keys():
                    break

                x_range, y_range = self.objs['可升级'][0]
                if not self.quite: print("click item for upgrade")
                self.random_click(x_range, y_range)
                # 再次点击工人，取消升级列表
                self.random_click((x1, x2), (y1, y2))
                self.update_objs()
                if '可升级' in self.objs.keys():
                    x_range, y_range = self.objs['可升级'][0]
                    self.random_click(x_range, y_range)
                else:
                    if self.click_txt("升级") is None:
                        self.click_txt("确认")
                    self.click_txt("确认")
                self.random_wait(0.3)
                self.update_objs()
                if '可升级' in self.objs.keys():
                    x_range, y_range = self.objs['可升级'][0]
                    self.random_click(x_range, y_range)
                else:
                    if self.click_txt("升级") is None:
                        self.click_txt("确认")
                    self.click_txt("确认")
                self.reset()
                break

    def remove_zw(self):
        self.reset()
        zw = '战魂'
        self.update_objs()
        if zw in self.objs.keys():
            x_range, y_range = self.objs[zw][0]
            self.random_click(x_range, y_range)

        zw = '障碍物'
        self.reset()
        cur_page = self.classify()[0]
        if not self.is_upgrader_available('工人'): return None
        self.d().pinch_in(50,20)
        # self.d().pinch_out(6, 10)
        l = n = m = 0
        fail_rm = 0
        if cur_page == "home_zsj": directions = ["right","left", "up", "down", "down", "down", "right", None]
        if cur_page == "home_ysj": directions = ["right","left", "left", "up", "down", "down", "right",
                                                 'right','down', "down", "right", "down","left", "up", None]

        for direction in directions:
            while True:
                self.reset()
                self.update_objs()
                time.sleep(1)
                if zw not in self.objs.keys(): break

                if not self.is_upgrader_available('工人'):
                    m += 1
                    if m > 5: break
                    time.sleep(15)
                else:
                    m = 0
                if zw in self.objs.keys():
                    x_range, y_range = self.objs[zw][0]
                    self.random_click(x_range, y_range)
                    time.sleep(0.3)
                    self.random_wait(0.1)
                    if self.click_obj('升级_btn') is not None:
                        n += 1
                        fail_rm = 0
                    elif self.click_txt('移除') is not None:
                        n += 1
                    else:
                        fail_rm += 1
                        print("fail_rm: ", fail_rm)
                        if fail_rm > 5:
                            break
                    time.sleep(0.2)
                    self.update_objs()
                    if '消耗宝石' in self.objs.keys():
                        self.click_obj('关闭', self.objs)
                        l += 1
                        fail_rm += 1
                        if l > 3: return None
                    time.sleep(10)
                    self.random_wait(1)
                self.update_objs()
                self.collectandacclerate()
            if direction is not None:
                self.d().swipe(direction, steps=50)

    def collectandacclerate(self):
        self.d().pinch_in(50,20)
        time.sleep(0.2)
        self.update_objs()
        for obj in ['圣水','金币','黑油','金币_y','宝石','遗留战利品','圣水_y']:
            if obj in self.objs.keys():
                x_range, y_range = self.objs[obj][0]
                self.random_click(x_range, y_range)
                time.sleep(0.3)
                if obj == '遗留战利品':
                    self.d().pinch_out(50,20)
                    self.random_wait(0.2)
                    self.click_obj('遗留战利品')
                    self.d().pinch_in(50, 20)

        self.update_objs()
        if '收集' in self.objs.keys():
            self.click_obj('收集', self.objs)
            time.sleep(0.2)
            self.click_obj('收集')
            time.sleep(0.2)
            self.click_obj('关闭')

        # 学徒工加速
        obj = '学徒工'
        if obj in self.objs.keys():
            # 第一次点击学徒工
            self.click_obj(obj, self.objs)

            # 第二次点击学徒工
            if self.click_obj(obj) is None:
                self.click_txt('指派')
                self.click_txt('工人')
                self.click_txt('建筑')

            if self.click_obj('增援') is None: self.click_txt('指派') # 指派学徒工 # 指派 的标注信息是 增援
            if self.click_obj('确定') is None: self.click_txt('确认') # 最后确认
            self.click_obj('关闭')
            self.update_objs()
        obj = '加速'
        if obj in self.objs.keys():
            self.click_obj(obj, self.objs)
            self.random_wait(1)
            if self.click_txt("免费提速") is None:
                self.click_txt('提速')
            self.random_wait(1)
            self.click_obj('确定')


    def segment(self, **kwargs):
        self.update_img()
        self.send_poses = self.yolo.segment(self.img, **kwargs)
        return self.send_poses


    def send_soldier(self, update_soldier=True, soldier=None, t=0.1, send_pos=None):
        # if self.classify()
        if update_soldier is True:
            self.update_objs()

        # 选择士兵
        if soldier is None:
            avs = []
            for i in self.objs.keys():
                if i.endswith("9"): avs.append(i)
            if len(avs) >= 1: soldier = random.choice(avs)
            else: return "AllOut"

        if soldier[:2] in ["男王","女王","闰土","永王","幽王"]: t=0.1
        if soldier.find("法术") > -1:
            t = 0.1
            send_pos = [random.randrange(max(self.res)), random.randrange(min(self.res))]
        if soldier in self.objs.keys():
            self.click_obj(soldier, objs=self.objs)
        else:
            print(soldier, "is not aviliable")
            return False

        # 选择区域并释放
        if send_pos is None:
            send_poses = self.segment()
            if send_poses is None:
                return False
            index = random.randint(0, len(send_poses[0]))
            send_pos = (send_poses[0][index],send_poses[1][index])
        self.clickpos(send_pos, t)
        self.update_stats()
        # if self.stats['已派出所有兵力'] > -1: break
        if self.stats['红线'] > -1:
            send_poses = self.segment()
            index = random.randint(0, len(send_poses[0]))
            send_pos = (send_poses[0][index],send_poses[1][index])
            time.sleep(5)

            self.send_soldier(update_soldier=False, soldier=soldier, t=t, send_pos=send_pos)
        return soldier


    def fight_yolo(self, condition="water > 50000 and gold > 50000 and oil > 0", strategy=None, detail=False,world=None):
        """

        :param condition:
        :param strategy:
        :param detail:
        :return:
        True: finished
        False: error
        -1: need soldier
        """
        if detail > 0 or not self.quite:
            detail = True
        self.update_cur_page()
        if self.cur_page not in ["home_zsj", "home_ysj", "counting"]:
            self.reset()
            self.update_cur_page()
        n=1
        while self.cur_page == "waiting":
            time.sleep(3)
            print("waiting...", end="")
            self.update_cur_page()
            n+=1;
            if n > 5:
                self.reset()
                return None

        if self.cur_page == "home_zsj":
            world = "zsj"
            r = self.zhao_yu(condition=condition)
            if r == -1: return -1

        elif self.cur_page == "home_ysj":
            world="ysj"
            self.update_objs()
            objs = self.objs
            if not self.click_obj("进攻_胜利之星", objs): self.click_obj("进攻", objs)
            time.sleep(0.5)
            self.click_obj("立即寻找")
            self.random_wait(5)
        elif self.cur_page == "counting":
            pass
        else:
            return None

        # 等待开始
        n=0
        while True:
            self.update_cur_page()
            if self.cur_page in ["warning", "counting"]: break
            time.sleep(3)
            n+=1
            if n > 10: break
        self.d().pinch_in(50,20)
        # war_start
        self.war_time_s=time.time()
        if strategy is None:
            self.update_objs()
            if detail is True: print("释放英雄", end=":")
            for i in ["男王9","女王9","闰土9","永王9","幽王9", "h9"]:
                if i in self.objs.keys():
                    self.send_soldier(1,soldier=i)
                    if detail is True: print(i, end=".")
                    self.random_wait(0.3)

            # 派出所有兵力
            if detail is True: print("\r释放英雄", end=".")
            if detail is True: print("释放所有兵力", end=":")
            while True:
                if time.time() - self.war_time_s > self.wait_time_after_release_s and world == "ysj":
                    self.reset()
                r = self.send_soldier(t=random.choice([0.1, random.randrange(100, 400) / 100]))
                if r == "AllOut": break
                else:
                    if detail is True: print(r, end=".")

            if detail is True: print("Wait for end", end="")
            wait = 5
            while True:
                # 释放技能
                if detail is True: print("技能:", end="")
                self.update_objs(all=True)
                for i in self.objs.keys():
                    if time.time() - self.war_time_s > self.wait_time_after_release_s and world == "ysj":
                        if not self.click_obj("放弃"): self.click_obj("结束战斗")
                        time.sleep(1)
                        self.click_obj("确定")
                        time.sleep(1)
                        self.click_obj("回营")
                        time.sleep(1)
                    if i.endswith("1") or i.endswith("2") or i.endswith("3"):
                        if detail is True: print(i, end=".")
                        self.click_obj(i, objs=self.objs)
                        self.random_wait(1)
                        break

                if time.time() - self.war_time_s > self.wait_time_after_release_s and world == "ysj":
                    self.reset()

                self.update_cur_page()
                if self.cur_page=="war_end":
                    if detail is True: print("\rend")
                    self.reset()
                    return "War end"
                elif self.cur_page=="counting":
                    # 夜世界2次战斗
                    self.fight_yolo(world="ysj")
                elif self.cur_page == "waring":
                    continue
                else:
                    self.reset()
                    break
                time.sleep(5)
        return True


if __name__ == '__main__':
    # if time.time() - 1732639604 < 6000:
    # time.sleep(120)
    # opt.build_zsj()
    #
    zsj_first=True
    waitTime = 100
    water=40000
    gold=40000
    oil=5000
    # runtimes = {'老机吧大':(1,9),
    #           '随法':(1,9),
    #           '泊松':(1,9),
    #           '私域':(1,9),
    #           '源代码':(1,9)}

    try:
        id = "7TJ7WKFA7PWGY5X4"  # 手机
        opt = Env(device_id=id, quite=True, wait_time_after_release_s=waitTime, detail=0)
        Users = ['老机吧大', '随法', '泊松', '私域', '源代码']
        # Users = [ '老机吧大', '私域']
        conditions = {'老机吧大': "water > 500000 and gold > 500000 or oil > 12000",
                      '随法': "water > 300000 and gold > 300000",
                      '泊松': "water > 300000 and gold > 300000",
                      '私域': "water > 300000 and gold > 300000",
                      '源代码': "water > 800000 and gold > 800000 or oil > 10000"}
    except:
        id = "127.0.0.1:5555"
        opt = Env(device_id=id, quite=True, wait_time_after_release_s=waitTime, detail=0, screens=1)
        Users=["金枪鱼","鲶鱼","草鱼"]
        conditions={'金枪鱼':"water > 1000 and gold > 1000 or oil > 12000",
                    '随法': "water > 300000 and gold > 300000",
                    '泊松': "water > 300000 and gold > 300000",
                    '私域': "water > 300000 and gold > 300000",
                    '鲶鱼':"water > 1000 and gold > 1000",
                    '草鱼':"water > 1000 and gold > 1000"}
    # opt.classify()
    # print(opt.yolo.cls_model.names)
    # print(opt.yolo.cls_model.names())
    # exit()
    opt.auto_fight_multi_users(users=Users, conditions=conditions,ysj_jg_times=5)
    # opt.all_day_long(users=Users, runtimes=runtimes, conditions=conditions, world = "both", zsj_first = zsj_first)
    # opt.all_day_long(screens=4,conditions ="water > 800000 and gold > 800000", runtimes=(3,7), zsj_first=True) # 马服

    print("Ending...")
    opt.exit()

# sleep 160
# yolo detect train data=data/COC_Images_V2/COC.yaml model=./tools/yolo11m.pt epochs=100 imgsz=1216 batch=1 name=yolo11COC1216m300ep verbose=True workers=0 close_mosaic=30 cos_lr=True
# sleep 160
# yolo detect train data=data/COC_Images_V2/COC.yaml model=./tools/yolo11m.pt epochs=100 imgsz=1600 batch=1 name=yolo11COC1600m300ep verbose=True workers=0 close_mosaic=30 cos_lr=True
# sleep 160
# yolo detect train data=data/COC_Images_V2/COC.yaml model=./tools/yolo11m.pt epochs=100 imgsz=1440 batch=1 name=yolo11COC1440m300ep verbose=True workers=0 close_mosaic=30 cos_lr=True
# sleep 160
# yolo detect train data=data/COC_Images_V2/COC.yaml model=./tools/yolo11m.pt epochs=300 imgsz=1024 batch=1 name=yolo11COC1024m300ep verbose=True workers=0 close_mosaic=30 exist_ok=True multi_scale=True  cos_lr=True
# sleep 160
