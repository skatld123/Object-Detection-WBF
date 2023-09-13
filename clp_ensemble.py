import argparse
import json
import os
from glob import glob
from pathlib import Path

import mmcv
import numpy as np
import pandas as pd
from ensemble_boxes import *
from ensemble_boxes.ensemble_boxes_nms import nms_method
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS
from natsort import natsorted
from tqdm import tqdm
from ultralytics import YOLO

def init_models(config_list, checkpoint_file_list) :
    model_list = []
    idx_yolo = []
    idx_mmdet = []
    if len(config_list) != len(checkpoint_file_list) : 
        print("config, checkpoint list의 길이가 일치하지 않습니다.")
        return
    else :
        for idx, (config, cf) in enumerate(zip(config_list, checkpoint_file_list)) :
            if 'yolo' in cf : 
                model = YOLO(cf)
                idx_yolo.append(idx)
                model_list.append(model)
            else :
                model = init_detector(config, cf, device='cuda:0')
                idx_mmdet.append(idx)
                model_list.append(model)
        return model_list, idx_yolo, idx_mmdet
    
def inference_models(model_list : list, data_path, threshold=0.5, yolo_idx=None, mmdet_idx=None, conf_thr=0.25) : 
    file_result_list = {}
    for idx, model in enumerate(model_list) :
        for i, file in enumerate(tqdm(natsorted(glob(data_path + '*')))):
            if idx in yolo_idx :
                file_name = Path(file).stem
                result = model(file, imgsz=1280, iou=threshold, conf=0.25)
                output = result[0].boxes
                boxes, conf, cls = output.xyxyn.tolist(), output.conf.tolist(), output.cls.tolist()
                if file_name not in file_result_list:
                    file_result_list[file_name] = [{"boxes_list":boxes, "score_list":conf, "label_list":cls}]
                else : file_result_list[file_name].append({"boxes_list":boxes, "score_list":conf, "label_list":cls})
            else :
                file_name = Path(file).stem
                img = mmcv.imread(file)
                ori_h, ori_w, _ = img.shape
                scale_factor = [ori_w, ori_h, ori_w, ori_h]
                result = inference_detector(model, img)
                
                result = result.pred_instances
                boxes_list = result.bboxes.tolist()
                score_list = result.scores.tolist()
                labels_list = result.labels.tolist()
                
                nms_boxes_list = []
                nms_score_list = []
                nms_labels_list = []
                for (box_, score_, label_) in zip(boxes_list, score_list, labels_list) :
                    if score_ >= conf_thr :
                        scaled_bbox = [coord / scale for coord, scale in zip(box_, scale_factor)]
                        nms_boxes_list.append(scaled_bbox)
                        nms_score_list.append(score_)
                        nms_labels_list.append(label_)
                if nms_boxes_list :
                    nms_boxes_list, nms_score_list, nms_labels_list = nms_method([nms_boxes_list], [nms_score_list], [nms_labels_list], weights=None, iou_thr=0.5, method=2)
                else : continue

                if file_name not in file_result_list :
                    file_result_list[file_name] = [{"boxes_list":nms_boxes_list.tolist(), "score_list":nms_score_list.tolist(), "label_list":nms_labels_list.tolist()}]
                else : file_result_list[file_name].append({"boxes_list":nms_boxes_list.tolist(), "score_list":nms_score_list.tolist(), "label_list":nms_labels_list.tolist()})
        print("file length : " + str(len(file_result_list)))
    return file_result_list

def ensemble_result(config_list : list[str], checkpoint_file_list : list[str], data_path=None, save_dir=None, iou_thr=0.55, skip_box_thr=0.0001, sigma = 0.1, weights=None) :
    if len(config_list) > 0 and len(checkpoint_file_list) > 0:
        model_list, idx_yolo, idx_mmdet = init_models(config_list, checkpoint_file_list)
        
        output_dic = inference_models(model_list, data_path=data_path, threshold=0.5, yolo_idx=idx_yolo, mmdet_idx=idx_mmdet)
            
        result_annotation = {}
        for idx, fileName in enumerate(output_dic) : 
            # output file
            f = open(os.path.join(save_dir, fileName + ".txt"), "w+")
            
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
            boxes, scores, labels = weighted_boxes_fusion(wbf_box_list, wbf_score_list, wbf_label_list, weights=weights, iou_thr=iou_thr, skip_box_thr=0.3)
            # Filtering
            nms_boxes_list = []
            nms_score_list = []
            nms_labels_list = []
            for (box_, score_, label_) in zip(boxes, scores, labels) :
                if score_ >= 0.25 :
                    nms_boxes_list.append(box_.tolist())
                    nms_score_list.append(score_.tolist())
                    nms_labels_list.append(label_.tolist())
            # result_annotation[fileName] = {"boxes":boxes.tolist(), "scores":scores.tolist(), "labels":labels.tolist()}
            result_annotation[fileName] = {"boxes":nms_boxes_list, "scores":nms_score_list, "labels":nms_labels_list}
            for box, score, label in zip(nms_boxes_list, nms_score_list, nms_labels_list) :
                box_str = ' '.join(str(round(coord,4)) for coord in box)
                f.write(str(label) + " " + str(round(score,4)) + " " + box_str +"\n")
            f.close()
            json_dir = os.path.abspath(os.path.join(save_dir, os.pardir))
        with open((json_dir + '/result.json'), 'w') as json_file:
            json.dump(result_annotation, json_file)
            
# if __name__ == '__main__':
#     if len(config_list) > 0 and len(checkpoint_file_list) > 0:
#         model_list, yolo_idx, mmdet_idx = init_models(config_list, checkpoint_file_list)
        
#         output_dic = inference_models(model_list, data_path=test_img_prefix, threshold=0.5, yolo_idx=yolo_idx, mmdet_idx=mmdet_idx)
        
#         iou_thr = 0.55
#         skip_box_thr = 0.0001
#         sigma = 0.1
#         result_annotation = {}
#         for idx, fileName in enumerate(output_dic) : 
#             # output file
#             f = open("/root/De-identification-CLP/ensemble_model/result/predict/" + fileName + ".txt", "w+")

#             wbf_box_list = []
#             wbf_score_list = []
#             wbf_label_list = []
#             for model_output in output_dic[fileName] :
#                 box_list = model_output["boxes_list"]  
#                 score_list = model_output["score_list"]
#                 label_list = model_output["label_list"]
#                 wbf_box_list.append(box_list)
#                 wbf_score_list.append(score_list)
#                 wbf_label_list.append(label_list)
#             boxes, scores, labels = weighted_boxes_fusion(wbf_box_list, wbf_score_list, wbf_label_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
#             result_annotation[fileName] = {"boxes":boxes.tolist(), "scores":score_list, "labels":labels.tolist()}
#             for box, score, label in zip(boxes.tolist(), scores, labels.tolist()) :
#                 box_str = ' '.join(str(round(coord,4)) for coord in box)
#                 f.write(str(label) + " " + str(round(score,4)) + " " + box_str +"\n")
#             f.close()
#         with open('/root/De-identification-CLP/ensemble_model/result/result.json', 'w') as json_file:
#             json.dump(result_annotation, json_file)