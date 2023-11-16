import argparse
import json
import os
from glob import glob
from pathlib import Path

import mmcv
import numpy as np
import pandas as pd
from ensemble_boxes import *
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS
from natsort import natsorted
from tqdm import tqdm
from clp_ensemble import init_models, inference_models, ensemble

# Specify the path to model config and checkpoint file
config_file_1 = '/root/mmdetection/configs/clp/swin_clp.py'
# config_file_2 = '/root/mmdetection/configs/clp/effb3_fpn_8xb4-crop896-1x_clp.py'
# config_file_3 = '/root/mmdetection/configs/clp/dcnv2_clp.py'
config_list = [config_file_1]

checkpoint_file_1 = '/root/mmdetection/work_dirs/swin_2044_200/epoch_100.pth'
# checkpoint_file_2 = '/root/mmdetection/work_dirs/effb3_2044_200/epoch_100.pth'
# checkpoint_file_3 = '/root/mmdetection/work_dirs/dcnv2_2044_200/epoch_100.pth'
checkpoint_file_list = [checkpoint_file_1]

weights = [1]

test_img_prefix = '/root/dataset_clp/dataset_2044/test/images/'

dir_prefix = ''

final_list = []
wbf_list = []
img_boxes_list = []
img_score_list = []
img_labels_list = []


if __name__ == '__main__':
    if len(config_list) > 0 and len(checkpoint_file_list) > 0:
        # 1차 모델 검출 -> 평가 
        # 2차 모델 검출 -> 평가
        # 앙상블 결과 -> 평가
        model_list = init_models(config_list, checkpoint_file_list)
        output_dic = inference_models(model_list, data_path=test_img_prefix)
        
        iou_thr = 0.55
        skip_box_thr = 0.0001
        sigma = 0.1
        result_annotation = {}
        for idx, fileName in enumerate(output_dic) : 
            # output file
            f = open("/root/ensemble_model/result/predict/" + fileName + ".txt", "w+")
            
            wbf_box_list = []
            wbf_score_list = []
            wbf_label_list = []
            for model_output in output_dic[fileName] :
                box_list = model_output["boxes_list"]  
                score_list = model_output["score_list"]
                label_list = model_output["label_list"]
                wbf_box_list.append(box_list)
                wbf_score_list.append(score_list)
                wbf_label_list.append(label_list)
            boxes, scores, labels = weighted_boxes_fusion(wbf_box_list, wbf_score_list, wbf_label_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
            result_annotation[fileName] = {"boxes":boxes.tolist(), "scores":score_list, "labels":labels.tolist()}
            for box, score, label in zip(boxes.tolist(), scores, labels.tolist()) :
                box_str = ' '.join(str(round(coord,4)) for coord in box)
                f.write(str(label) + " " + str(round(score,4)) + " " + box_str +"\n")
            f.close()