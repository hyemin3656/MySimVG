import time
import torch
import numpy
import copy

import pycocotools.mask as maskUtils
from simvg.datasets import extract_data
from simvg.utils import get_root_logger, reduce_mean, is_main
from torchvision.ops.boxes import box_area
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
from collections import defaultdict


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

    mask_iou = torch.tensor([0.0], device=device)
    mask_acc_at_thrs = torch.full((5,), -1.0, device=device)
    if eval_mask:
        mask_iou = mask_overlaps(gt_mask, pred_masks, is_crowd).to(device)
        for i, iou_thr in enumerate([0.5, 0.6, 0.7, 0.8, 0.9]):
            mask_acc_at_thrs[i] = (mask_iou >= iou_thr).float().mean()

    return det_acc * 100.0, mask_iou * 100.0, mask_acc_at_thrs * 100.0


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


def evaluate_model(epoch, cfg, model, loader, train_loader=None, writer=None):
    model.eval()

    device = list(model.parameters())[0].device

    batches = len(loader)
    end = time.time()

    with_bbox, with_mask = False, False
    det_acc_list, mask_iou_list, mask_acc_list, f1_score_list, n_acc_list = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    with torch.no_grad():
        total_loss= defaultdict(float)
        total_sample = 0
        exis_total_loss = defaultdict(float)
        nt_all = {
            "TP": torch.tensor(0.0, device=device),
            "TN": torch.tensor(0.0, device=device),
            "FP": torch.tensor(0.0, device=device),
            "FN": torch.tensor(0.0, device=device),
        }
        correct_image_all = 0
        num_image_all = 0
        dummy_dict_all = {
            'num_dummy': torch.tensor(0.0, device=device),
            'num_no_target' : torch.tensor(0.0, device=device),
            'num_accurate_dummy': torch.tensor(0.0, device=device)
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

            if not cfg.distributed:
                inputs = extract_data(inputs)
            inputs["epoch"]=epoch
            inputs["batch"] = batch
            inputs["batches"] = batches
            losses_dict, predictions, dummy_dict, dev = model(
                **inputs,
                train_flag=False,
                return_loss=True,
                rescale=False,
                #with_bbox=with_bbox,
                #with_mask=with_mask,
            )
            
            # 누적
            for key in selected_keys:
                value = losses_dict[key]
                total_loss[key] += value.item() * batch_sample #batch_sample로 수정하기
                
            sample_size = [batch_sample, dummy_dict['num_no_target'], batch_sample-dummy_dict['num_no_target']]
            loss_exis_score = losses_dict.pop("loss_exis_score", torch.tensor([0.0], device=device))
            for (key, size) in zip(exis_keys, sample_size):
                if size > 0:
                    value = loss_exis_score[key]
                    exis_total_loss[key] += value.item() * size
                
            if not isinstance(predictions, list):
                predictions_list = [predictions]
            else:
                predictions_list = predictions

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
                    det_acc_list[predict_type].append(batch_det_acc.item())
                    det_acc = sum(det_acc_list[predict_type]) / len(det_acc_list[predict_type])
                    det_acc_dict[predict_type] = det_acc
                else:
                    targets = [meta["target"] for meta in img_metas]
                    with torch.no_grad():
                        batch_f1_score, batch_n_acc, nt, correct_image, num_image = grec_evaluate_f1_nacc(pred_bboxes, gt_bbox, targets, device=device)
                        nt_all['TP']+=nt['TP']
                        nt_all['TN']+=nt['TN']
                        nt_all['FP']+=nt['FP']
                        nt_all['FN']+=nt['FN']
                        correct_image_all+=correct_image
                        num_image_all+=num_image
                        if cfg.distributed:
                            batch_f1_score = reduce_mean(batch_f1_score)
                            batch_n_acc = reduce_mean(batch_n_acc)
                    f1_score_list[predict_type].append(batch_f1_score.item())
                    n_acc_list[predict_type].append(batch_n_acc.item())
                    f1_score_acc = sum(f1_score_list[predict_type]) / len(f1_score_list[predict_type])
                    n_acc = sum(n_acc_list[predict_type]) / len(n_acc_list[predict_type])
                    f1_score_acc_dict[predict_type] = f1_score_acc
                    n_acc_dict[predict_type] = n_acc
                    #dummy
                    for key in dummy_dict_all.keys():
                        dummy_dict_all[key] += dummy_dict[key]

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
                    F1_Score_str = "".join(F1_Score_str_list)
                    n_acc_str = "".join(n_acc_str_list)
                    logger.info(
                        f"Validate - epoch [{epoch+1}]-[{batch+1}/{batches}] "
                        + f"time: {(time.time()- end):.2f}, "
                        + F1_Score_str
                        + n_acc_str
                        + f"num_dummy: {dummy_dict['num_dummy'].item()}"
                        + f"num_no_target: {dummy_dict['num_no_target'].item()}"
                    )
            

            end = time.time()
        #전체 val detection loss
        avg_loss_dict = {k: v / total_sample for k, v in total_loss.items()}
        print(avg_loss_dict)
        #전체 N-acc
        N_acc_all = nt_all["TP"] / (nt_all["TP"] + nt_all["FN"]) if nt_all["TP"] != 0 else torch.tensor(0.0, device=device)
        N_acc_all = N_acc_all.float() * 100
        print('N_acc_all', N_acc_all)
        #전체 F1
        F1_score_all = correct_image_all / num_image_all
        F1_score_all = F1_score_all.float() * 100
        print('F1_score_all', F1_score_all)
        #Tensorboard
        if writer is not None:
            #val loss
            if train_loader is not None:
                x_step = (epoch+1) * len(train_loader)
            print(x_step)
            writer.add_scalars(f"Loss/val", avg_loss_dict, x_step) 
            
            #전체 exis loss
            sample_sizes = [total_sample, dummy_dict_all['num_no_target'], total_sample-dummy_dict_all['num_no_target']]
            avg_exis_loss_dict = {k:v / sample_size for (k, v), sample_size in zip(exis_total_loss.items(), sample_sizes)}
            writer.add_scalars(f"Exis_Loss/val", avg_exis_loss_dict, x_step)
            #val F1, N-acc
            writer.add_scalars(f"f1", {"val_f1":F1_score_all.item()}, x_step)
            writer.add_scalars(f"N-acc", {"val_N-acc":N_acc_all.item()}, x_step)

            print("val_f1:", F1_score_all)
            print("val_N-acc:", N_acc_all)

            #전체 Dummy precision
            dummy_precision_all = dummy_dict_all['num_accurate_dummy']/dummy_dict_all['num_dummy']
            writer.add_scalars(f"dummy_val", {"val_dummy_precision":dummy_precision_all.item()}, x_step)
            #전체 Dummy recall
            dummy_recall_all = dummy_dict_all['num_accurate_dummy']/dummy_dict_all['num_no_target']
            writer.add_scalars(f"dummy_val", {"val_dummy_recall":dummy_recall_all.item()}, x_step)
            dummy_f1_all = 2*(dummy_precision_all*dummy_recall_all)/(dummy_precision_all+dummy_recall_all)
            writer.add_scalars(f"dummy_val", {"val_dummy_f1":dummy_f1_all.item()}, x_step)
            writer.add_scalars(f"dummy_num_val", {"val_dummy_num":dummy_dict_all['num_dummy'].item()}, x_step)
            print('dummy_num', dummy_dict_all['num_dummy'].item(), 'dummy_precision_all', dummy_precision_all.item(), 'dummy_recall_all', dummy_recall_all.item(), 'dummy_f1_all', dummy_f1_all.item())

            #dev
            if dev is not None:
                writer.add_scalars(f"dev/val/no_target", {
                    'dotpro':  dev['no_target']['dotpro']/dummy_dict_all['num_no_target'],
                    'scaled':  dev['no_target']['scaled']/dummy_dict_all['num_no_target']
                }, x_step)
                writer.add_scalars(f"dev/val/others", {
                    'dotpro':  dev['others']['dotpro']/ (total_sample - dummy_dict_all['num_no_target']),
                    'scaled':  dev['others']['scaled']/ (total_sample - dummy_dict_all['num_no_target'])
                }, x_step)
                                   
    if not cfg["dataset"] == "GRefCOCO":
        det_acc = sum(list(det_acc_dict.values())) / len(det_acc_dict)
        mask_iou = 0
    else:
        det_acc = sum(list(f1_score_acc_dict.values())) / len(f1_score_acc_dict)
        mask_iou = sum(list(n_acc_dict.values())) / len(n_acc_dict)
        

    return det_acc, mask_iou
