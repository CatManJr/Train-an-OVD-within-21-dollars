import os
import codecs
import numpy as np
import math
import cv2
import shapely.geometry as shgeo
import copy
import time

def choose_best_pointorder_fit_another(poly1, poly2):
    """
    给定两个多边形，重排第一个多边形的点顺序，使两者更为匹配
    （主要针对多边形，但目前主要用处在 DOTA 中，水平框可不用该函数）
    """
    x1, y1 = poly1[0], poly1[1]
    x2, y2 = poly1[2], poly1[3]
    x3, y3 = poly1[4], poly1[5]
    x4, y4 = poly1[6], poly1[7]
    combinate = [
        np.array([x1, y1, x2, y2, x3, y3, x4, y4]),
        np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
        np.array([x3, y3, x4, y4, x1, y1, x2, y2]),
        np.array([x4, y4, x1, y1, x2, y2, x3, y3])
    ]
    dst_coordinate = np.array(poly2)
    distances = np.array([np.sum((coord - dst_coordinate)**2) for coord in combinate])
    sorted_indices = distances.argsort()
    return combinate[sorted_indices[0]]

def cal_line_length(point1, point2):
    """
    计算两点间的欧氏距离
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

class splitbase():
    def __init__(self,
                 basepath,
                 outpath,
                 code='utf-8',
                 gap=100,
                 subsize=1024,
                 thresh=0.7,
                 choosebestpoint=True,
                 ext='.png'):
        """
        初始化参数：
        - basepath: 数据集根目录，内部包含 images 与 labelTxt 文件夹
        - outpath: 输出目录，分割图片与标签将保存在 images 与 labelTxt 中
        - code: 读取/写入标签文件的编码格式
        - gap: 子图间重叠部分尺寸
        - subsize: 子图尺寸（假设为正方形）
        - thresh: 当目标只部分落入子图时，交集面积/原目标面积的阈值
        - choosebestpoint: 针对多边形目标排序的选择开关（水平框可保持 False）
        - ext: 图像扩展名
        """
        self.basepath = basepath
        self.outpath = outpath
        self.code = code
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.thresh = thresh
        self.choosebestpoint = choosebestpoint
        self.ext = ext
        self.imagepath = os.path.join(self.basepath, 'image')
        self.labelpath = os.path.join(self.basepath, 'label')
        self.outimagepath = os.path.join(self.outpath, 'image')
        self.outlabelpath = os.path.join(self.outpath, 'label')
        os.makedirs(self.outimagepath, exist_ok=True)
        os.makedirs(self.outlabelpath, exist_ok=True)

    def polyorig2sub(self, left, up, bbox):
        """
        将水平目标框（[xmin, ymin, xmax, ymax]）的原始坐标转换到子图内的相对坐标
        """
        return [bbox[0] - left, bbox[1] - up, bbox[2] - left, bbox[3] - up]

    def calchalf_iou(self, bbox, sub_box):
        """
        利用 Shapely 计算两个矩形（水平框）的交集，
        返回交集多边形及其交集面积与原目标面积的比值
        """
        poly1 = shgeo.box(*bbox)
        poly2 = shgeo.box(*sub_box)
        # 如果原始框面积为 0，则直接返回 0
        if poly1.area == 0:
            return poly1.intersection(poly2), 0
        inter_poly = poly1.intersection(poly2)
        half_iou = inter_poly.area / poly1.area
        return inter_poly, half_iou

    def saveimagepatches(self, img, subimgname, left, up):
        """
        根据当前窗口坐标从原图中剪切子图并保存
        """
        subimg = copy.deepcopy(img[up: (up + self.subsize), left: (left + self.subsize)])
        outdir = os.path.join(self.outimagepath, subimgname + self.ext)
        cv2.imwrite(outdir, subimg)

    def GetPoly4FromPoly5(self, poly):
        """
        当切割结果为五边形时，通过两个短边的中点生成新的四边形
        （主要用于 DOTA 的倾斜框，此处水平框可不使用）
        """
        distances = [cal_line_length((poly[i * 2], poly[i * 2 + 1]),
                                     (poly[(i + 1) * 2], poly[(i + 1) * 2 + 1]))
                     for i in range(int(len(poly) / 2 - 1))]
        distances.append(cal_line_length((poly[0], poly[1]), (poly[8], poly[9])))
        pos = np.array(distances).argsort()[0]
        count = 0
        outpoly = []
        while count < 5:
            if count == pos:
                outpoly.append((poly[count * 2] + poly[(count * 2 + 2) % 10]) / 2)
                outpoly.append((poly[(count * 2 + 1) % 10] + poly[(count * 2 + 3) % 10]) / 2)
                count += 1
            elif count == (pos + 1) % 5:
                count += 1
                continue
            else:
                outpoly.append(poly[count * 2])
                outpoly.append(poly[count * 2 + 1])
                count += 1
        return outpoly

    def savepatches(self, img, objects, subimgname, left, up, right, down):
        """
        遍历所有标注对象，根据分割窗口与目标框的交集情况：
         • 若交集等于整个目标（half_iou == 1），直接转换原始框坐标；
         • 若交集部分大于0且比例超过 thresh，则计算交集区域后转换坐标；
         • 否则，舍弃该目标。
        最后将生成的 YOLO 格式标签写入子图对应的 txt 中，并保存子图图像。
        如果子图像素和为零，则跳过生成 txt 标签和保存子图像。
        """
        outdir = os.path.join(self.outlabelpath, subimgname + '.txt')
        sub_box = [left, up, right, down]
        with codecs.open(outdir, 'w', self.code) as f_out:
            for obj in objects:
                bbox = obj['bbox']  # [xmin, ymin, xmax, ymax]
                _, half_iou = self.calchalf_iou(bbox, sub_box)
                if half_iou == 1:
                    new_bbox = self.polyorig2sub(left, up, bbox)
                elif half_iou >= self.thresh:
                    new_xmin = max(bbox[0], left)
                    new_ymin = max(bbox[1], up)
                    new_xmax = min(bbox[2], right)
                    new_ymax = min(bbox[3], down)
                    new_bbox = [new_xmin - left, new_ymin - up, new_xmax - left, new_ymax - up]
                else:
                    continue

                # 将新的框转换为 YOLO 格式 (归一化)
                sub_w = right - left
                sub_h = down - up
                x_center = (new_bbox[0] + new_bbox[2]) / 2.0 / sub_w
                y_center = (new_bbox[1] + new_bbox[3]) / 2.0 / sub_h
                box_w = (new_bbox[2] - new_bbox[0]) / sub_w
                box_h = (new_bbox[3] - new_bbox[1]) / sub_h
                f_out.write(f"{obj['name']} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
        self.saveimagepatches(img, subimgname, left, up)

    def SplitSingle(self, name, rate, extent):
        """
        读取单张原图和对应标签（YOLO 格式），对图像进行缩放（若 rate !=1），
        然后按照 slide 步长在图像上滑动生成子图，每个子图调用 savepatches 生成和保存对应标注。
        """
        img = cv2.imread(os.path.join(self.imagepath, name + extent))
        if img is None:
            return
        fullname = os.path.join(self.labelpath, name + '.txt')
        objects = []
        if os.path.exists(fullname):
            with codecs.open(fullname, 'r', self.code) as f_in:
                for line in f_in:
                    parts = line.strip().split()
                    # YOLO 格式：name cx cy w h (归一化)
                    if len(parts) == 5:
                        obj_name = parts[0]
                        cx, cy, w_, h_ = map(float, parts[1:])
                        H, W = img.shape[:2]
                        x_center = cx * W
                        y_center = cy * H
                        box_w = w_ * W
                        box_h = h_ * H
                        bbox = [x_center - box_w/2, y_center - box_h/2, x_center + box_w/2, y_center + box_h/2]
                        objects.append({'bbox': bbox, 'name': obj_name})
        if rate != 1:
            resizeimg = cv2.resize(img, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
        else:
            resizeimg = img
        outbasename = name + '__' + str(rate) + '__'
        weight = resizeimg.shape[1]
        height = resizeimg.shape[0]

        left, up = 0, 0
        while (left < weight):
            if (left + self.subsize >= weight):
                left = max(weight - self.subsize, 0)
            up = 0
            while (up < height):
                if (up + self.subsize >= height):
                    up = max(height - self.subsize, 0)
                right = min(left + self.subsize, weight - 1)
                down = min(up + self.subsize, height - 1)
                subimgname = outbasename + str(left) + '___' + str(up)
                # self.f_sub.write(name + ' ' + subimgname + ' ' + str(left) + ' ' + str(up) + '\n')
                # 跳过全黑的无效图片
                subimg = resizeimg[up:down, left:right]
                if subimg.size != 0 and np.count_nonzero(subimg) != 0:
                    # Skip sub-images that are completely black or empty
                    self.savepatches(resizeimg, objects, subimgname, left, up, right, down)
                if (up + self.subsize >= height):
                    break
                else:
                    up = up + self.slide
            if (left + self.subsize >= weight):
                break
            else:
                left = left + self.slide

    def splitdata(self, rate):
        """
        遍历图像目录，对每张图像应用 SplitSingle 方法完成数据分割
        """
        img_list = os.listdir(self.imagepath)
        imagenames = [os.path.splitext(x)[0] for x in img_list if os.path.splitext(x)[1].lower() == self.ext.lower()]
        for name in imagenames:
            self.SplitSingle(name, rate, self.ext)

if __name__ == '__main__':

    start_time = time.time()
    
    # example usage of splitbase (处理水平框，YOLO 格式标签)
    split = splitbase(
        basepath='/Volumes/T7 Shield/xView/xView',
        outpath='/Volumes/T7 Shield/xView/split',
        code='utf-8',
        gap=200,
        subsize=1024,
        thresh=0.7,
        choosebestpoint=False,
        ext='.png'
    )
    split.splitdata(1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Done! Processed in {elapsed_time:.2f} seconds.")
    print(f"Processed {len(os.listdir(split.imagepath))} images.")