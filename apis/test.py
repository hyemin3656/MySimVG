import time
import torch
import numpy
import numpy as np
import copy

import pycocotools.mask as maskUtils
from simvg.datasets import extract_data
from simvg.utils import get_root_logger, reduce_mean, is_main
from torchvision.ops.boxes import box_area
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
from collections import defaultdict
import torch.nn.functional as F



def mask_overlaps(gt_mask, pred_masks, is_crowd):
    """Args:
    gt_mask (list[RLE]):
    pred_mask (list[RLE]):
    """

    def computeIoU_RLE(gt_mask, pred_masks, is_crowd):
        mask_iou = maskUtils.iou(pred_masks, gt_mask, is_crowd)
        mask_iou = numpy.diag(mask_iou)
        return mask_iou

    mask_iou = computeIoU_RLE(gt_mask, pred_masks, is_crowd)
    mask_iou = torch.from_numpy(mask_iou)

    return mask_iou


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def accuracy(pred_bboxes, gt_bbox, pred_masks, gt_mask, is_crowd=None, device="cuda:0"):
    eval_det = pred_bboxes is not None
    eval_mask = pred_masks is not None

    det_acc = torch.tensor([0.0], device=device)
    bbox_iou = torch.tensor([0.0], device=device)
    if eval_det:
        gt_bbox = torch.stack(gt_bbox).to(device)
        bbox_iou = bbox_overlaps(gt_bbox, pred_bboxes, is_aligned=True)
        det_acc = (bbox_iou >= 0.5).float().mean()
        num_correct_ot = (bbox_iou >= 0.5).float().sum()

    mask_iou = torch.tensor([0.0], device=device)
    mask_acc_at_thrs = torch.full((5,), -1.0, device=device)
    if eval_mask:
        mask_iou = mask_overlaps(gt_mask, pred_masks, is_crowd).to(device)
        for i, iou_thr in enumerate([0.5, 0.6, 0.7, 0.8, 0.9]):
            mask_acc_at_thrs[i] = (mask_iou >= iou_thr).float().mean()

    return det_acc * 100.0, mask_iou * 100.0, mask_acc_at_thrs * 100.0, num_correct_ot


def grec_evaluate_f1_nacc(predictions, gt_bboxes, targets, thresh_score=0.7, thresh_iou=0.5, thresh_F1=1.0, device="cuda:0"):
    #predictions = list(dict) #이미지 마다 박스좌표, object 예측확률, 클래스 레이블 저장 #len(predictions) = bs
    correct_image = torch.tensor(0, device=device)
    num_image = torch.tensor(0, device=device)
    nt = {
        "TP": torch.tensor(0.0, device=device),
        "TN": torch.tensor(0.0, device=device),
        "FP": torch.tensor(0.0, device=device),
        "FN": torch.tensor(0.0, device=device),
    }
    if predictions is None:
        return torch.tensor(0.0, device=device).float(), torch.tensor(0.0, device=device).float()
    for prediction, gt_bbox, target in zip(predictions, gt_bboxes, targets): #각 샘플마다 순회
        TP = 0
        assert prediction is not None
        #score(object 예측 확률) 기준 내림차순 정렬
        sorted_scores_boxes = sorted(zip(prediction["scores"].tolist(), prediction["boxes"].tolist()), reverse=True)
        if len(sorted_scores_boxes) == 0:
            print(prediction["scores"]) #(n_q)
            print(prediction["boxes"]) #(n_q, 4)
        sorted_scores, sorted_boxes = zip(*sorted_scores_boxes) #(n_q), #(n_q, 4)
        sorted_boxes = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes])
        converted_bbox_all = []
        no_target_flag = False
        for converted_bbox, one_target in zip(gt_bbox, target): #GT box 수만큼 순회 (many target일 경우 여러 번)
            if one_target["category_id"] == -1: #해당 샘플이 NO-target 샘플일 경우
                no_target_flag = True
            # target_bbox = one_target["bbox"]
            # converted_bbox = [
            #     target_bbox[0],
            #     target_bbox[1],
            #     target_bbox[2] + target_bbox[0],
            #     target_bbox[3] + target_bbox[1],
            # ]
            converted_bbox_all.append(converted_bbox)
        gt_bbox_all = torch.stack(converted_bbox_all, dim=0) #no-target/single-target : (1, 4) or many-target : (2, 4)

        sorted_scores_array = numpy.array(sorted_scores)
        idx = sorted_scores_array >= thresh_score
        filtered_boxes = sorted_boxes[idx]
        # filtered_boxes = sorted_boxes[0:1]
        giou = generalized_box_iou(filtered_boxes, gt_bbox_all.view(-1, 4)) #(num_prediction, num_gt)
        num_prediction = filtered_boxes.shape[0] #score기준 필터링한 후 남은 예측 박스(쿼리) 수
        num_gt = gt_bbox_all.shape[0] # 1 or 2
        if no_target_flag:
            if num_prediction >= 1: #1개 이상이라도 object class 예측확률이 0.7 이상인 쿼리가 있는 경우
                nt["FN"] += 1
                F_1 = torch.tensor(0.0, device=device)
            else:
                nt["TP"] += 1
                F_1 = torch.tensor(1.0, device=device)
        else:
            if num_prediction >= 1:
                nt["TN"] += 1
            else:
                nt["FP"] += 1
            for i in range(min(num_prediction, num_gt)):
                top_value, top_index = torch.topk(giou.flatten(0, 1), 1) #max giou의 (predict, gt) 박스 쌍
                if top_value < thresh_iou: 
                    break
                else:
                    top_index_x = top_index // num_gt
                    top_index_y = top_index % num_gt
                    TP += 1 #매칭 한 쌍을 하나의 TP로 간주
                    #이미 매칭된 건 초기화
                    giou[top_index_x[0], :] = 0.0 
                    giou[:, top_index_y[0]] = 0.0
            #모든 GT박스마다 threshold giou 넘는 예측박스들이 존재할 경우에만 F_1 = 1 되도록
            #num_prediction == num_gt 이면서 thres_iou 넘는 각 매칭 쌍이 모두 존재할 때 F_1 = 1
            FP = num_prediction - TP
            FN = num_gt - TP
            F_1 = 2 * TP / (2 * TP + FP + FN)

        if F_1 >= thresh_F1:
            correct_image += 1
        num_image += 1

    batch_F1_score = correct_image / num_image #F1 score가 1인 이미지들의 비율
    # T_acc = nt["TN"] / (nt["TN"] + nt["FP"])
    batch_N_acc = nt["TP"] / (nt["TP"] + nt["FN"]) if nt["TP"] != 0 else torch.tensor(0.0, device=device)
    return batch_F1_score.float() * 100, batch_N_acc.float() * 100, nt, correct_image, num_image

def grec_evaluate_f1_nacc_detacc(predictions, gt_bboxes, targets, thresh_score=0.7, thresh_iou=0.5, thresh_F1=1.0, device="cuda:0"):
    #predictions = list(dict) #이미지 마다 박스좌표, object 예측확률, 클래스 레이블 저장 #len(predictions) = bs
    correct_image = torch.tensor(0, device=device)
    num_image = torch.tensor(0, device=device)
    nt = {
        "TP": torch.tensor(0.0, device=device),
        "TN": torch.tensor(0.0, device=device),
        "FP": torch.tensor(0.0, device=device),
        "FN": torch.tensor(0.0, device=device),
    }
    if predictions is None:
        return torch.tensor(0.0, device=device).float(), torch.tensor(0.0, device=device).float()
    one_target_mask = torch.ones(len(predictions), dtype=torch.bool, device=device)
    top_1_boxes = []
    num_nt = 0
    #confidence 체크
    filtered_conf_nt = []
    filtered_conf_ot = []
    filtered_conf_mt = []
    for i, (prediction, gt_bbox, target) in enumerate(zip(predictions, gt_bboxes, targets)): #각 샘플마다 순회
        TP = 0
        assert prediction is not None
        #score(object 예측 확률) 기준 내림차순 정렬
        sorted_scores_boxes = sorted(zip(prediction["scores"].tolist(), prediction["boxes"].tolist()), reverse=True)
        if len(sorted_scores_boxes) == 0:
            print(prediction["scores"]) #(n_q)
            print(prediction["boxes"]) #(n_q, 4)
        sorted_scores, sorted_boxes = zip(*sorted_scores_boxes) #(n_q), #(n_q, 4)
        top_1_boxes.append(torch.tensor(sorted_boxes[0], device=device).unsqueeze(0)) #각 샘플마다 가장 높은 확률의 예측 박스 저장
        sorted_boxes = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes])
        converted_bbox_all = []
        no_target_flag = False
        for converted_bbox, one_target in zip(gt_bbox, target): #GT box 수만큼 순회 (many target일 경우 여러 번)
            if one_target["category_id"] == -1: #해당 샘플이 NO-target 샘플일 경우
                no_target_flag = True
                one_target_mask[i] = False
                num_nt += 1
            # target_bbox = one_target["bbox"]
            # converted_bbox = [
            #     target_bbox[0],
            #     target_bbox[1],
            #     target_bbox[2] + target_bbox[0],
            #     target_bbox[3] + target_bbox[1],
            # ]
            converted_bbox_all.append(converted_bbox)
        if len(converted_bbox_all) >= 2 : #many-target인 경우(3개 이상일 수도 있음)
            one_target_mask[i] = False
        gt_bbox_all = torch.stack(converted_bbox_all, dim=0) #no-target/single-target : (1, 4) or many-target : (2, 4)

        sorted_scores_array = numpy.array(sorted_scores)
        idx = sorted_scores_array >= thresh_score
        filtered_boxes = sorted_boxes[idx] # All Dummy) 빈 텐서
        # filtered_boxes = sorted_boxes[0:1]
        giou = generalized_box_iou(filtered_boxes, gt_bbox_all.view(-1, 4)) #(num_prediction, num_gt) #All Dummy) 빈 텐서 반환
        num_prediction = filtered_boxes.shape[0] #score기준 필터링한 후 남은 예측 박스(쿼리) 수
        #confidence check
        # if no_target_flag == True:
        #     filtered_conf_nt.append(num_prediction)
        # elif one_target_mask[i] == True:
        #     filtered_conf_ot.append(num_prediction)
        # else:
        #     filtered_conf_mt.append(num_prediction)

        
        num_gt = gt_bbox_all.shape[0] # 1 or 2
        if no_target_flag:
            if num_prediction >= 1: #1개 이상이라도 object class 예측확률이 0.7 이상인 쿼리가 있는 경우
                nt["FN"] += 1
                F_1 = torch.tensor(0.0, device=device)
            else:
                nt["TP"] += 1
                F_1 = torch.tensor(1.0, device=device)
        else:
            if num_prediction >= 1:
                nt["TN"] += 1
            else:
                nt["FP"] += 1
            for i in range(min(num_prediction, num_gt)):
                top_value, top_index = torch.topk(giou.flatten(0, 1), 1) #max giou의 (predict, gt) 박스 쌍
                if top_value < thresh_iou: 
                    break
                else:
                    top_index_x = top_index // num_gt
                    top_index_y = top_index % num_gt
                    TP += 1 #매칭 한 쌍을 하나의 TP로 간주
                    #이미 매칭된 건 초기화
                    giou[top_index_x[0], :] = 0.0 
                    giou[:, top_index_y[0]] = 0.0
            #모든 GT박스마다 threshold giou 넘는 예측박스들이 존재할 경우에만 F_1 = 1 되도록
            #num_prediction == num_gt 이면서 thres_iou 넘는 각 매칭 쌍이 모두 존재할 때 F_1 = 1
            FP = num_prediction - TP
            FN = num_gt - TP
            F_1 = 2 * TP / (2 * TP + FP + FN)

        if F_1 >= thresh_F1:
            correct_image += 1
        num_image += 1

    #confidence check
    def print_conf_stats(conf_list, name):
        if len(conf_list) > 0:
            mean = np.mean(conf_list)
            var = np.var(conf_list)
            print(conf_list)
            print(f"{name} - 평균: {mean:.2f}, 분산: {var:.2f}")
        else:
            print(f"{name} - 데이터가 없음")

    #print_conf_stats(filtered_conf_nt, "No Target (filtered_conf_nt)")
    #print_conf_stats(filtered_conf_ot, "One Target (filtered_conf_ot)")
    #print_conf_stats(filtered_conf_mt, "Multiple Targets (filtered_conf_mt)")
    #print('-----------------')
    
    batch_F1_score = correct_image / num_image #F1 score가 1인 이미지들의 비율
    # T_acc = nt["TN"] / (nt["TN"] + nt["FP"])
    batch_N_acc = nt["TP"] / (nt["TP"] + nt["FN"]) if nt["TP"] != 0 else torch.tensor(0.0, device=device)
    
    num_ot = one_target_mask.sum()
    #pre@(IOU>=0.5)
    if one_target_mask.any():
        pred_bboxes = torch.cat(top_1_boxes, dim=0) #(bs, 4)
        ot_pred_bboxes = pred_bboxes[one_target_mask, :] #(num_one_target, 4)
        ot_gt_bboxes = [bbox.squeeze(0) for bbox, keep in zip(gt_bboxes, one_target_mask) if keep]
        batch_det_acc, _, _, num_correct_ot = accuracy(ot_pred_bboxes, ot_gt_bboxes, None, None)
    else:
        batch_det_acc = None
        num_correct_ot = 0

    return batch_F1_score.float() * 100, batch_N_acc.float() * 100, batch_det_acc, nt, correct_image, num_image, num_ot, num_correct_ot
def evaluate_model(epoch, cfg, model, loader, train_loader=None, writer=None):
    model.eval()

    device = list(model.parameters())[0].device

    batches = len(loader)
    batches_for_check_under = batches
    batches_for_check_over = batches
    end = time.time()

    with_bbox, with_mask = False, False
    det_acc_list, mask_iou_list, mask_acc_list, f1_score_list, n_acc_list, loss_det_list = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list)
    )
    avg_loss_dict_ = defaultdict(float)
    num_not_all_dummy = torch.tensor(0.0, device=device)
    with torch.no_grad():
        topk_cos_sim_loss_flag = False
        total_loss= defaultdict(float)
        total_sample = 0
        exis_total_loss = {  # defaultdict 대신 dict로
            'loss_score_mean': 0.0,
            'no_target_los_mean': 0.0,
            'others_los_mean': 0.0,
        }
        more_than_ten_target = 0
        num_no_target_all = 0 #
        num_ot_all = 0
        num_mt_all = 0
        nt_topk_sum_all = 0
        ot_topk_sum_all = 0
        mt_topk_sum_all = 0
        nt_all = {
            "TP": torch.tensor(0.0, device=device),
            "TN": torch.tensor(0.0, device=device),
            "FP": torch.tensor(0.0, device=device),
            "FN": torch.tensor(0.0, device=device),
        }
        correct_image_all = 0
        num_image_all = 0
        num_ot_all = 0
        num_correct_ot_all = 0
        dummy_dict_all = {
            'num_all_dummy': torch.tensor(0.0, device=device),
            'num_accurate_dummy': torch.tensor(0.0, device=device),
            'sum_dummy_ratio_of_part_dum_nt': torch.tensor(0.0, device=device),
            'sum_part_dummy_of_nt': torch.tensor(0.0, device=device),
            'sum_dummy_ratio_of_part_dum_others': torch.tensor(0.0, device=device),
            'sum_part_dummy_of_others': torch.tensor(0.0, device=device),
            'ratio_over_cross_blah': torch.tensor(0.0, device=device),
            'ratio_under_cross_blah': torch.tensor(0.0, device=device)
        }
        selected_keys = ['loss_class', 'loss_bbox', 'loss_giou', 'loss_det']
        exis_keys = ['loss_score_mean', 'no_target_los_mean', 'others_los_mean']
        for batch, inputs in enumerate(loader):
            gt_bbox, gt_mask, is_crowd = None, None, None
            batch_sample = len(inputs["gt_bbox"].data[0])
            total_sample += batch_sample
            if "gt_bbox" in inputs:
                with_bbox = True
                if isinstance(inputs["gt_bbox"], torch.Tensor):
                    inputs["gt_bbox"] = [inputs["gt_bbox"][ind] for ind in range(inputs["gt_bbox"].shape[0])]
                    gt_bbox = copy.deepcopy(inputs["gt_bbox"])
                else:
                    gt_bbox = copy.deepcopy(inputs["gt_bbox"].data[0])
                    
            if "gt_mask_rle" in inputs:
                with_mask = True
                gt_mask = inputs.pop("gt_mask_rle").data[0]
            if "is_crowd" in inputs:
                is_crowd = inputs.pop("is_crowd").data[0]

            img_metas = inputs["img_metas"].data[0]
            #-----------
            no_target = torch.zeros(batch_sample, dtype=torch.bool, device=device)
            for i, meta in enumerate(img_metas):
                for target in meta['target']:
                    if target['category_id']==-1:
                        no_target[i]=True
            num_no_target = no_target.sum()
            num_no_target_all += num_no_target
            #-----------
            ot_bool = torch.zeros(batch_sample, dtype=torch.bool, device=device)
            mt_bool = torch.zeros(batch_sample, dtype=torch.bool, device=device)
            num_ot = 0
            for i, (bbox_one_sample) in enumerate(gt_bbox):
                if not no_target[i]:
                    if bbox_one_sample.shape[0] >10:
                        more_than_ten_target += 1
                    if bbox_one_sample.shape[0] == 1:
                        ot_bool[i]=True
                        num_ot += 1
                    if bbox_one_sample.shape[0] >= 2:
                        mt_bool[i]=True
            num_mt = (batch_sample - num_no_target) - num_ot
            num_ot_all += num_ot
            num_mt_all += num_mt
                        
            if not cfg.distributed:
                inputs = extract_data(inputs)
            inputs["epoch"]=epoch
            inputs["batch"] = batch
            inputs["batches"] = batches
            losses_dict, predictions, topk_per_batch_mean, dummy_dict, dev = model( #basline은 dummy_dict, dev가 None
                **inputs,
                train_flag=False,
                return_loss=True,
                rescale=False,
                #with_bbox=with_bbox,
                #with_mask=with_mask,
            )
            
                    
            batch_sample_size = [batch_sample, num_no_target, batch_sample-num_no_target]
            if "loss_exis_score" in losses_dict:
                exis_enc_flag = True
                loss_exis_score = losses_dict.pop("loss_exis_score")
                for key, size in zip(exis_keys, batch_sample_size):
                    if size > 0:
                        value = loss_exis_score[key]
                        exis_total_loss[key] += value.item() * size
            else:
                exis_enc_flag = False
            dummy_token_diversity_loss = losses_dict.pop("dummy_token_diversity_loss", torch.tensor([0.0], device=device))
            nt_dummy_loss = losses_dict.pop("nt_dummy_loss", torch.tensor([0.0], device=device))
            others_dummy_loss = losses_dict.pop("others_dummy_loss", torch.tensor([0.0], device=device))

            if dummy_dict:
                dummy_idx = dummy_dict['dummy_idx']
                min_num_f1_0_mt_sample = 0
                min_num_f1_0_ot_sample = 0
                for i, (bbox_one_sample, dummy_idx_one_sample) in enumerate(zip(gt_bbox, dummy_idx)):
                    if not no_target[i]:
                        if bbox_one_sample.shape[0] > (~dummy_idx_one_sample).sum():
                            if bbox_one_sample.shape[0]==1:
                                min_num_f1_0_ot_sample += 1
                            if bbox_one_sample.shape[0]>1:
                                min_num_f1_0_mt_sample += 1
                min_num_f1_0_mt_sample_ratio = min_num_f1_0_mt_sample / num_mt if num_mt > 0 else float('nan')
                print(min_num_f1_0_mt_sample_ratio)
                min_num_f1_0_ot_sample_ratio = min_num_f1_0_ot_sample/num_ot if num_ot>0 else float('nan')
                print(min_num_f1_0_ot_sample_ratio)
                
            if exis_enc_flag:
                nt_topk_sum = topk_per_batch_mean[no_target].sum()
                nt_topk_sum_all += nt_topk_sum
                ot_topk_sum = topk_per_batch_mean[ot_bool].sum()
                ot_topk_sum_all += ot_topk_sum
                mt_topk_sum = topk_per_batch_mean[mt_bool].sum()
                mt_topk_sum_all += mt_topk_sum

                gt_topk_mean = ~no_target
                gt_topk_mean = gt_topk_mean.float()
                if topk_cos_sim_loss_flag:
                    topk_cos_sim_loss = F.mse_loss(topk_per_batch_mean, gt_topk_mean, reduction="mean")
                else: topk_cos_sim_loss = 0
                    
            if not isinstance(predictions, list):
                predictions_list = [predictions]
            else:
                predictions_list = predictions
                
            for loss_name, loss_value in losses_dict.items():
                if loss_name in selected_keys:
                    if cfg.distributed:
                        loss_value = reduce_mean(loss_value)
                    loss_det_list[loss_name].append(loss_value.item())
                
            # statistics informations
            map_dict = {0: "decoder", 1: "token"}
            det_acc_dict, f1_score_acc_dict, n_acc_dict = {}, {}, {}
            for ind, predictions in enumerate(predictions_list):
                predict_type = map_dict[ind]
                pred_bboxes = predictions.pop("pred_bboxes")
                pred_masks = predictions.pop("pred_masks")
                if not cfg["dataset"] == "GRefCOCO":
                    with torch.no_grad():
                        batch_det_acc, batch_mask_iou, batch_mask_acc_at_thrs = accuracy(
                            pred_bboxes,
                            gt_bbox,
                            pred_masks,
                            gt_mask,
                            is_crowd=is_crowd,
                            device=device,
                        )
                        if cfg.distributed:
                            batch_det_acc = reduce_mean(batch_det_acc)
                            # batch_mask_iou = reduce_mean(batch_mask_iou)
                            # batch_mask_acc_at_thrs = reduce_mean(batch_mask_acc_at_thrs)
                    det_acc = sum(det_acc_list[predict_type]) / len(det_acc_list[predict_type])
                    det_acc_dict[predict_type] = det_acc
                else:
                    targets = [meta["target"] for meta in img_metas]
                    with torch.no_grad():
                        #batch_f1_score, batch_n_acc, nt, correct_image, num_image = grec_evaluate_f1_nacc(pred_bboxes, gt_bbox, targets, device=device)
                        batch_f1_score, batch_n_acc, batch_det_acc, nt, correct_image, num_image, num_ot, num_correct_ot=grec_evaluate_f1_nacc_detacc(pred_bboxes, gt_bbox, targets, device=device)
                        nt_all['TP']+=nt['TP']
                        nt_all['TN']+=nt['TN']
                        nt_all['FP']+=nt['FP']
                        nt_all['FN']+=nt['FN']
                        correct_image_all+=correct_image
                        num_image_all+=num_image
                        num_correct_ot_all+=num_correct_ot
                        num_ot_all+=num_ot
                        if cfg.distributed:
                            batch_f1_score = reduce_mean(batch_f1_score)
                            batch_n_acc = reduce_mean(batch_n_acc)
                    f1_score_list[predict_type].append(batch_f1_score.item())
                    n_acc_list[predict_type].append(batch_n_acc.item())
                    if batch_det_acc!=None:
                        det_acc_list[predict_type].append(batch_det_acc.item())
                    f1_score_acc = sum(f1_score_list[predict_type]) / len(f1_score_list[predict_type])
                    n_acc = sum(n_acc_list[predict_type]) / len(n_acc_list[predict_type])
                    det_acc = sum(det_acc_list[predict_type]) / len(det_acc_list[predict_type])
                    f1_score_acc_dict[predict_type] = f1_score_acc
                    n_acc_dict[predict_type] = n_acc
                    det_acc_dict[predict_type] = det_acc
                    #dummy
                    if dummy_dict is not None:
                        for key in dummy_dict_all.keys():
                            if dummy_dict[key] != None:
                                dummy_dict_all[key] += dummy_dict[key]
                            else:
                                if key == "ratio_over_cross_blah":
                                    batches_for_check_over -= 1 
                                if key == "ratio_under_cross_blah":
                                    batches_for_check_under -= 1 

            # logging informations
            if is_main() and ((batch + 1) % cfg.log_interval == 0 or batch + 1 == batches):
                logger = get_root_logger()

                if not cfg["dataset"] == "GRefCOCO":
                    ACC_str_list = [
                        "{}Det@.5: {:.2f}, ".format(map_dict[i], det_acc_dict[map_dict[i]]) for i in range(len(predictions_list))
                    ]
                    ACC_str = "".join(ACC_str_list)
                    logger.info(f"val - epoch [{epoch+1}]-[{batch+1}/{batches}] " + f"time: {(time.time()- end):.2f}, " + ACC_str)
                    
                else:
                    F1_Score_str_list = [
                        "{}_f1_score: {:.2f}, ".format(map_dict[i], f1_score_acc_dict[map_dict[i]]) for i in range(len(predictions_list))
                    ]
                    n_acc_str_list = [
                        "{}_n_acc: {:.2f}, ".format(map_dict[i], n_acc_dict[map_dict[i]]) for i in range(len(predictions_list))
                    ]
                    ACC_str_list = [
                        "{}Acc:{:.2f}, ".format(map_dict[i], det_acc_dict[map_dict[i]]) for i in range(len(predictions_list))
                    ]
                    F1_Score_str = "".join(F1_Score_str_list)
                    n_acc_str = "".join(n_acc_str_list)
                    ACC_str = "".join(ACC_str_list)
                    logger.info(
                        f"Validate - epoch [{epoch+1}]-[{batch+1}/{batches}] "
                        + f"time: {(time.time()- end):.2f}, "
                        + F1_Score_str
                        + n_acc_str
                        + ACC_str
                        # + f"num_dummy: {dummy_dict['num_all_dummy'].item()}"
                        # + f"num_no_target: {dummy_dict['num_no_target'].item()}"
                    )
            

            end = time.time()
            
        #전체 N-acc
        N_acc_all = nt_all["TP"] / (nt_all["TP"] + nt_all["FN"]) if nt_all["TP"] != 0 else torch.tensor(0.0, device=device)
        N_acc_all = N_acc_all.float() * 100
        #전체 F1
        F1_score_all = correct_image_all / num_image_all
        F1_score_all = F1_score_all.float() * 100
        #전체 det acc
        det_acc_all = num_correct_ot_all/num_ot_all
        det_acc_all = det_acc_all.float()*100
        print("more_than_ten_target", more_than_ten_target)
        #Tensorboard
        if writer is not None:
            if train_loader is not None:
                x_step = (epoch+1) * len(train_loader)
                
            #val loss
            for loss_n, loss_v in loss_det_list.items():
                loss_n = f"{loss_n.split('loss_')[-1]}"
                avg_loss = sum(loss_v)/len(loss_v)
                avg_loss_dict_[loss_n] = avg_loss
            writer.add_scalars(f"Loss/val", avg_loss_dict_, x_step)
            #val F1, N-acc, det_acc
            writer.add_scalars(f"metric/f1", {"val_f1":F1_score_all.item()}, x_step)
            writer.add_scalars(f"metric/N-acc", {"val_N-acc":N_acc_all.item()}, x_step)
            writer.add_scalars(f"metric/det_acc", {"val_det_acc":det_acc_all.item()}, x_step)
            
            #전체 exis loss
            if exis_enc_flag :
                sample_sizes = [total_sample, num_no_target_all, total_sample-num_no_target_all]
                avg_exis_loss_dict = {k:v / sample_size for (k, v), sample_size in zip(exis_total_loss.items(), sample_sizes)}
                writer.add_scalars(f"Exis_Loss/val", avg_exis_loss_dict, x_step)
                print(sample_sizes)

                Exis_distinguish = {"nt": nt_topk_sum_all/num_no_target_all, "ot": ot_topk_sum_all/num_ot_all, "mt": mt_topk_sum_all/num_mt_all}
                writer.add_scalars(f"Exis_distinguish/val", Exis_distinguish, x_step)
                if topk_cos_sim_loss_flag:
                    writer.add_scalars(f"topk_cos_sim_loss_flag", {"val":topk_cos_sim_loss.item()}, x_step)

            if dummy_dict is not None:
                #해당 배치의 diversity loss
                writer.add_scalar(f"dummy_diversity_loss/val", dummy_token_diversity_loss, x_step)
                #해당 배치의 dummy enhance loss
                writer.add_scalar(f"dummy_enhance_loss/val/nt", nt_dummy_loss, x_step)
                writer.add_scalar(f"dummy_enhance_loss/val/others", others_dummy_loss, x_step)

                print(f"val_f1: {F1_score_all.item()}, val_N-acc: {N_acc_all.item()}, val_det_acc: {det_acc_all.item()}")

                #전체 Dummy precision
                dummy_precision_all = dummy_dict_all['num_accurate_dummy']/dummy_dict_all['num_all_dummy']
                writer.add_scalars(f"dummy_metric/val", {"dummy_precision":dummy_precision_all.item()}, x_step)
                #전체 Dummy recall
                dummy_recall_all = dummy_dict_all['num_accurate_dummy']/num_no_target_all
                writer.add_scalars(f"dummy_metric/val", {"dummy_recall":dummy_recall_all.item()}, x_step)
                dummy_f1_all = 2*(dummy_precision_all*dummy_recall_all)/(dummy_precision_all+dummy_recall_all)
                writer.add_scalars(f"dummy_metric/val", {"dummy_f1":dummy_f1_all.item()}, x_step)
                writer.add_scalars(f"dummy_ratio/val", {"dummy_num/total_size":dummy_dict_all['num_all_dummy'].item()/sample_sizes[0]}, x_step)
                nt_denom = dummy_dict_all['sum_part_dummy_of_nt'].item()
                others_denom = dummy_dict_all['sum_part_dummy_of_others'].item()

                #min_num_f1_0_sample_ratio
                writer.add_scalars(f"min_num_f1_0_sample_ratio/val", {"min_num_f1_0_mt_sample_ratio":min_num_f1_0_mt_sample_ratio, "min_num_f1_0_ot_sample_ratio": min_num_f1_0_ot_sample_ratio}, x_step)

                if nt_denom > 1e-6 and others_denom > 1e-6:
                    #전체 dummy ratio
                    writer.add_scalars(f"extract_part_dummy/ratio/val", {"no-target":dummy_dict_all['sum_dummy_ratio_of_part_dum_nt'].item()/nt_denom, "others":dummy_dict_all['sum_dummy_ratio_of_part_dum_others'].item()/others_denom}, x_step)
                #part dummy 수
                writer.add_scalars(f"extract_part_dummy/num_sample/val", {"no-target": dummy_dict_all['sum_part_dummy_of_nt'].item(), "others":dummy_dict_all['sum_part_dummy_of_others'].item()}, x_step)
                
                #print('dummy_precision_all', dummy_precision_all.item(), 'dummy_recall_all', dummy_recall_all.item(), 'dummy_f1_all', dummy_f1_all.item())

                #dev
                if dev is not None:
                    writer.add_scalars(f"dev/val/no_target", {
                        'dotpro':  dev['no_target']['sim']/num_no_target_all,
                        'scaled':  dev['no_target']['scaled_sim']/num_no_target_all
                    }, x_step)
                    writer.add_scalars(f"dev/val/others", {
                        'dotpro':  dev['others']['sim']/ (total_sample - num_no_target_all),
                        'scaled':  dev['others']['scaled_sim']/ (total_sample - num_no_target_all)
                    }, x_step)
                #ratio_under/over_cross_blah
                writer.add_scalars(f"ratio_cross_blah/val", {
                    'yes_dum_no_target_over_cross' : dummy_dict_all['ratio_over_cross_blah']/batches_for_check_over,
                    'yes_dum_no_target_under_cross' : dummy_dict_all['ratio_under_cross_blah']/batches_for_check_under
                }, x_step)
                                   
    if not cfg["dataset"] == "GRefCOCO":
        det_acc = sum(list(det_acc_dict.values())) / len(det_acc_dict)
        mask_iou = 0
    else:
        det_acc = sum(list(f1_score_acc_dict.values())) / len(f1_score_acc_dict)
        mask_iou = sum(list(n_acc_dict.values())) / len(n_acc_dict)
        

    return det_acc, mask_iou
