import cv2
import matplotlib.pyplot as plt
import os
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from tqdm import tqdm
from collections import Counter

# num2class = {"0.0" : "license-plate", "1.0" : "vehicle"}
# # box : (centerX, centerY, width, height)
def convertToAbsoluteValues(size, box):
    xIn = round(((2 * float(box[0]) - float(box[2])) * size[0] / 2))
    yIn = round(((2 * float(box[1]) - float(box[3])) * size[1] / 2))
    xEnd = xIn + round(float(box[2]) * size[0])
    yEnd = yIn + round(float(box[3]) * size[1])
    if xIn < 0:
        xIn = 0
    if yIn < 0:
        yIn = 0
    if xEnd >= size[0]:
        xEnd = size[0] - 1
    if yEnd >= size[1]:
        yEnd = size[1] - 1
    return (xIn, yIn, xEnd, yEnd)

def convert_scaled_predict(size, box):
    scale_factor = size + size
    scaled_bbox = [round(coord * scale) for coord, scale in zip(box, scale_factor)]
    return scaled_bbox[0], scaled_bbox[1], scaled_bbox[2], scaled_bbox[3]

def boundingBoxes_fromDic(labelPath, prediction : dict, imgPath) :
    detections, groundtruths, classes = [], [], []
    labelList = os.listdir(labelPath)
    no_detection = []
    print("Read Annotations and Predicts files")
    for file in tqdm(labelList):
        file_path = os.path.join(labelPath, file)
        filename = os.path.splitext(file)[0]
        if filename in prediction :
            with open(file_path) as f:
                labelinfos = f.readlines()

            imgfilepath = os.path.join(imgPath, filename + ".jpg")
            img = cv.imread(imgfilepath)
            h, w, _ = img.shape

            for labelinfo in labelinfos:
                cls, rx1, ry1, rx2, ry2 = map(float, labelinfo.strip().split())
                x1, y1, x2, y2 = convertToAbsoluteValues((w, h), (rx1, ry1, rx2, ry2))
                boxinfo = [filename, cls, 1, (x1, y1, x2, y2)]
                if cls not in classes:
                    classes.append(cls)
                groundtruths.append(boxinfo)
            
            boxes = prediction[filename]['boxes']
            scores = prediction[filename]['scores']
            labels = prediction[filename]['labels']
            
            for box, conf, cls in zip(boxes, scores, labels) :
                rx1, ry1, rx2, ry2 = box
                x1, y1, x2, y2 = convert_scaled_predict((w, h), (rx1, ry1, rx2, ry2))
                boxinfo = [filename, cls, conf, (x1, y1, x2, y2)]
                if cls not in classes:
                    classes.append(cls)
                detections.append(boxinfo)
        else :
            no_detection.append(filename)
    classes = sorted(classes)
    print(f'List of objects not detected from Two-Stage Detection : {no_detection}')
    return detections, groundtruths, classes

# 라벨과 이미지 경로로부터 바운딩 박스 반환하기
def boundingBoxes(labelPath, predictPath, imagePath):
    detections, groundtruths, classes = [], [], []
    gt_and_prediect = [labelPath, predictPath]
    print("Read Annotations and Predict")
    for idx, type_dir in enumerate(gt_and_prediect) :
        if type_dir == labelPath : print("Read Annotation files")
        else : print("Read Prediction files")
        for file in tqdm(os.listdir(type_dir)):
            file_path = os.path.join(type_dir,file)
            filename = os.path.splitext(file)[0]
            
            with open(file_path) as f:
                labelinfos = f.readlines()

            imgfilepath = os.path.join(imagePath, filename + ".jpg")
            img = cv.imread(imgfilepath)
            h, w, _ = img.shape

            if type_dir == labelPath :
                for labelinfo in labelinfos:
                    cls, rx1, ry1, rx2, ry2 = map(float, labelinfo.strip().split())
                    x1, y1, x2, y2 = convertToAbsoluteValues((w, h), (rx1, ry1, rx2, ry2))
                    boxinfo = [filename, cls, 1, (x1, y1, x2, y2)]
                    if cls not in classes:
                        classes.append(cls)
                    groundtruths.append(boxinfo)
            else : 
                for labelinfo in labelinfos:
                    cls, conf, rx1, ry1, rx2, ry2 = map(float, labelinfo.strip().split())
                    x1, y1, x2, y2 = convert_scaled_predict((w, h), (rx1, ry1, rx2, ry2))
                    boxinfo = [filename, cls, conf, (x1, y1, x2, y2)]
                    if cls not in classes:
                        classes.append(cls)
                    detections.append(boxinfo)
                        
    classes = sorted(classes)
                
    return detections, groundtruths, classes


def boxPlot(boxlist, imagePath, savePath):
    labelfiles = sorted(list(set([filename for filename, _, _, _ in boxlist])))
    for labelfile in tqdm(labelfiles):
        rectinfos = []
        imgfilePath = os.path.join(imagePath, labelfile + ".jpg")
        img = cv.imread(imgfilePath)

        for filename, cls, conf, (x1, y1, x2, y2) in boxlist:
            if labelfile == filename:
                rectinfos.append((x1, y1, x2, y2, cls, conf))
                
        for x1, y1, x2, y2, cls_id, conf in rectinfos:
            
            if conf == 1.0:
                rectcolor = (0, 255, 0)
                cv.rectangle(img, (x1, y1), (x2, y2), rectcolor, 4)
            else:
                rectcolor = (0, 0, 255)
                text = f'cls:{cls_id}/conf:{conf:.2f}'
                cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv.rectangle(img, (x1, y1), (x2, y2), rectcolor, 4)
                
        cv.imwrite(f"{savePath}/{labelfile}.jpg", img)

def getArea(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def getUnionAreas(boxA, boxB, interArea=None):
    area_A = getArea(boxA)
    area_B = getArea(boxB)
    
    if interArea is None:
        interArea = getIntersectionArea(boxA, boxB)
        
    return float(area_A + area_B - interArea)

def getIntersectionArea(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # intersection area
    return (xB - xA + 1) * (yB - yA + 1)

def boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[2]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return False  # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[3]:
        return False  # boxA is below boxB
    return True

def iou(boxA, boxB):
    # if boxes dont intersect
    if boxesIntersect(boxA, boxB) is False:
        return 0
    interArea = getIntersectionArea(boxA, boxB)
    union = getUnionAreas(boxA, boxB, interArea=interArea)
    
    # intersection over union
    result = interArea / union
    assert result >= 0
    return result

def calculateAveragePrecision(rec, prec):
    
    mrec = [0] + [e for e in rec] + [1]
    mpre = [0] + [e for e in prec] + [0]

    for i in range(len(mpre)-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])

    ii = []

    for i in range(len(mrec)-1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i+1)

    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i-1]) * mpre[i])
    
    return [ap, mpre[0:len(mpre)-1], mrec[0:len(mpre)-1], ii]

def ElevenPointInterpolatedAP(rec, prec):

    mrec = [e for e in rec]
    mpre = [e for e in prec]

    recallValues = np.linspace(0, 1, 11)
    recallValues = list(recallValues[::-1])
    rhoInterp, recallValid = [], []

    for r in recallValues:
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0

        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min():])

        recallValid.append(r)
        rhoInterp.append(pmax)

    ap = sum(rhoInterp) / 11

    return [ap, rhoInterp, recallValues, None]

def AP(detections, groundtruths, classes, IOUThreshold = 0.5, method = 'AP', save_img=False, save_path=None):
    print("Save False Positive Img and Calcuating AP per Class")
    result = []
    for c in classes:

        dects = [d for d in detections if d[1] == c]
        gts = [g for g in groundtruths if g[1] == c]

        npos = len(gts)

        dects = sorted(dects, key = lambda conf : conf[2], reverse=True)

        TP = np.zeros(len(dects))
        FP = np.zeros(len(dects))

        det = Counter(cc[0] for cc in gts)

        # 각 이미지별 ground truth box의 수
        # {99 : 2, 380 : 4, ....}
        # {99 : [0, 0], 380 : [0, 0, 0, 0], ...}
        for key, val in det.items():
            det[key] = np.zeros(val)

        for d in tqdm(range(len(dects))):

            gt = [gt for gt in gts if gt[0] == dects[d][0]]

            iouMax = 0

            for j in range(len(gt)):
                iou1 = iou(dects[d][3], gt[j][3])
                if iou1 > iouMax:
                    iouMax = iou1
                    jmax = j
            # dects[d][0] -> filename, label, conf, (box)
            if iouMax >= IOUThreshold:
                if det[dects[d][0]][jmax] == 0:
                    TP[d] = 1
                    det[dects[d][0]][jmax] = 1
                else:
                    FP[d] = 1
                    # if save_img :
                    #     img = cv.imread(os.path.join('/root/dataset_clp/dataset_v2/test/images', dects[d][0] + ".jpg"))
                    #     x1, y1, x2, y2 = dects[d][3]
                    #     rectcolor = (0, 0, 255)
                    #     text = f'cls:{dects[d][1]}/conf:{round(dects[d][2], 2):.2f}'
                    #     cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, rectcolor, 2)
                    #     cv.rectangle(img, (x1, y1), (x2, y2), rectcolor, 4)
                    #     cv.imwrite(os.path.join(save_path, f"{d}_{round(dects[d][0], 2)}.jpg"))
                                
            else:
                FP[d] = 1
                if save_img :
                        img = cv.imread(os.path.join('/root/dataset_clp/dataset_v2/test/images', dects[d][0] + ".jpg"))
                        x1, y1, x2, y2 = dects[d][3]
                        rectcolor = (0, 0, 255)
                        text = f'cls:{dects[d][1]}/conf:{round(dects[d][2], 2):.2f}'
                        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, rectcolor, 2)
                        cv.rectangle(img, (x1, y1), (x2, y2), rectcolor, 4)
                        cv.imwrite(os.path.join(save_path, f"{dects[d][0]}_{round(dects[d][2], 2)}.jpg"), img)

        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        rec = acc_TP / npos
        prec = np.divide(acc_TP, (acc_FP + acc_TP))

        if method == "AP":
            [ap, mpre, mrec, ii] = calculateAveragePrecision(rec, prec)
        else:
            [ap, mpre, mrec, _] = ElevenPointInterpolatedAP(rec, prec)

        r = {
            'class' : c,
            'precision' : prec,
            'recall' : rec,
            'AP' : ap,
            'interpolated precision' : mpre,
            'interpolated recall' : mrec,
            'total positives' : npos,
            'total TP' : np.sum(TP),
            'total FP' : np.sum(FP)
        }

        result.append(r)

    return result

def mAP(result):
    ap = 0
    for r in result:
        ap += r['AP']
    mAP = ap / len(result)
    
    return mAP


def cal_mAP(num2class, detections, groundtruths, classes, save_img=False, save_path=None):
    # IoU
    boxA = detections[-1][-1]
    boxB = groundtruths[-1][-1]

    print(f"boxA coordinates : {(boxA)}")
    print(f"boxA area : {getArea(boxA)}")
    print(f"boxB coordinates : {(boxB)}")
    print(f"boxB area : {getArea(boxB)}")
    print(f"Union area of boxA and boxB : {getUnionAreas(boxA, boxB)}")
    print(f"Does boxes Intersect? : {boxesIntersect(boxA, boxB)}")
    print(
        f"Intersection area of boxA and boxB : {getIntersectionArea(boxA, boxB)}")
    print(f"IoU of boxA and boxB : {iou(boxA, boxB)}")

    result = AP(detections, groundtruths, classes, save_img=save_img, save_path=save_path)

    for r in result:
        print("{:^8} AP : {}".format(num2class[str(r['class'])], r['AP']))
    print("---------------------------")
    print(f"mAP : {mAP(result)}")
