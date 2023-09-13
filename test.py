import argparse
from glob import glob
import os
from pathlib import Path
import json 
import mmcv
import numpy as np
import pandas as pd
from mmdet.apis import inference_detector, init_detector
from natsort import natsorted
from tqdm import tqdm
from ensemble_boxes import *
from mmdet.registry import VISUALIZERS

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

def init_models(config_list, checkpoint_file_list) :
    model_list = []
    if len(config_list) != len(checkpoint_file_list) : 
        print("config, checkpoint list의 길이가 일치하지 않습니다.")
        return
    else :
        for (config, cf) in zip(config_list, checkpoint_file_list) :
            model = init_detector(config, cf, device='cuda:0')
            model_list.append(model)
        return model_list
    
def inference_models(model_list : list, data_path) : 
    file_result_list = {}
    threshold = 0.5
    for idx, model in enumerate(model_list) :
        for i, file in enumerate(tqdm(natsorted(glob(data_path + '*')))):
            file_name = Path(file).stem
            img = mmcv.imread(file)
            ori_h, ori_w, _ = img.shape
            scale_factor = [ori_w, ori_h, ori_w, ori_h]
            result = inference_detector(model, img)
            
            # visualizer = VISUALIZERS.build(model.cfg.visualizer)
            # visualizer.dataset_meta = model.dataset_meta
            # # # Show the results
            # visualizer.add_datasample(
            #     'result',
            #     img,
            #     data_sample=result,
            #     draw_gt=True,
            #     pred_score_thr=0.5,
            #     out_file='/root/ensemble_model/swin/'+file_name+".jpg"
            # )
            
            result = result.pred_instances
            boxes_list = result.bboxes.tolist()
            score_list = result.scores.tolist()
            labels_list = result.labels.tolist()
            nms_boxes_list = []
            nms_score_list = []
            nms_labels_list = []
            for (box_, score_, label_) in zip(boxes_list, score_list, labels_list) :
                if score_ >= threshold :
                    scaled_bbox = [coord / scale for coord, scale in zip(box_, scale_factor)]
                    nms_boxes_list.append(scaled_bbox)
                    nms_score_list.append(score_)
                    nms_labels_list.append(label_)
                    
            # scaled_bbox_list = []
            # for bbox in boxes_list :
            #     scaled_bbox = [coord / scale for coord, scale in zip(bbox, scale_factor)]
            #     scaled_bbox_list.append(scaled_bbox)

            # print(score_list)
            # print(labels_list)
            if idx == 0 :
                file_result_list[file_name] = [{"boxes_list":nms_boxes_list, "score_list":nms_score_list, "label_list":nms_labels_list}]
            else : file_result_list[file_name].append({"boxes_list":nms_boxes_list, "score_list":nms_score_list, "label_list":nms_labels_list})

        print("file length : " + str(len(file_result_list)))
    return file_result_list

if __name__ == '__main__':
    if len(config_list) > 0 and len(checkpoint_file_list) > 0:
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