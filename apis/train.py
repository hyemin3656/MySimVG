import time
import copy
import numpy
import torch
import random

#from simvg.apis.test import grec_evaluate_f1_nacc
from simvg.apis.test import grec_evaluate_f1_nacc_detacc
import torch.nn.functional as F


from .test import accuracy
from simvg.datasets import extract_data
from simvg.utils import get_root_logger, reduce_mean, is_main
from collections import defaultdict

try:
    import apex
except:
    pass


def set_random_seed(seed, deterministic=False):
    """Args:
    seed (int): Seed to be used.
    deterministic (bool): Whether to set the deterministic option for
        CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
        to True and `torch.backends.cudnn.benchmark` to False.
        Default: False.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(epoch, cfg, model, model_ema, optimizer, loader, writer=None):
    model.train()

    if cfg.distributed:
        loader.sampler.set_epoch(epoch)

    device = list(model.parameters())[0].device

    num_batches = len(loader)
    batches_for_check_over = num_batches
    batches_for_check_under = num_batches
    end = time.time()

    loss_det_list, det_acc_list, n_acc_list, f1_score_list = (defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list))
    avg_loss_dict_ = defaultdict(float)
    
    #epoch단위 N-acc, F1 관련 초기화---
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
    loss_weight_flag = True
    
    #-------------------------------
    topk_cos_sim_loss_flag=False
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
    loss_weight = {'det_loss': 1, 'exis_loss': 1}
    exis_total_loss = {  # defaultdict 대신 dict로
        'loss_score_mean': 0.0,
        'no_target_los_mean': 0.0,
        'others_los_mean': 0.0,
    }

    num_no_target_all = 0 #
    num_ot_all = 0
    num_mt_all = 0
    nt_topk_sum_all = 0
    ot_topk_sum_all = 0
    mt_topk_sum_all = 0
    total_sample = 0
    selected_keys = ['loss_class', 'loss_bbox', 'loss_giou', 'loss_det']
    exis_keys = ['loss_score_mean', 'no_target_los_mean', 'others_los_mean']
    more_than_ten_target = 0
    num_not_all_dummy = torch.tensor(0.0, device=device)
    #torch.autograd.set_detect_anomaly(True)
    for batch, inputs in enumerate(loader):
        data_time = time.time() - end
        gt_bbox, gt_mask, is_crowd = None, None, None
        batch_sample = len(inputs["gt_bbox"].data[0])
        total_sample += batch_sample
        if "gt_bbox" in inputs:
            if isinstance(inputs["gt_bbox"], torch.Tensor):
                inputs["gt_bbox"] = [inputs["gt_bbox"][ind] for ind in range(inputs["gt_bbox"].shape[0])]
                gt_bbox = copy.deepcopy(inputs["gt_bbox"])
            else:
                gt_bbox = copy.deepcopy(inputs["gt_bbox"].data[0])
            
        img_metas = inputs["img_metas"].data[0]
        no_target = torch.zeros(batch_sample, dtype=torch.bool, device=device)
        for i, meta in enumerate(img_metas):
            for target in meta['target']:
                if target['category_id']==-1:
                    no_target[i]=True
        num_no_target = no_target.sum()
        num_no_target_all += num_no_target

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
             
        if "gt_mask_rle" in inputs:
            gt_mask = inputs.pop("gt_mask_rle").data[0]
        if "is_crowd" in inputs:
            is_crowd = inputs.pop("is_crowd").data[0]

        if not cfg.distributed:
            inputs = extract_data(inputs)
        inputs["epoch"] = epoch
        inputs["batch"] = batch
        inputs["batches"] = num_batches #532

        losses_dict, predictions, topk_per_batch_mean, dummy_dict, dev = model(**inputs, rescale=False) #LGQS = False 일 경우 dummy_dict, dev는 None

        # #loss 누적
        # for key in selected_keys:
        #     if key in losses_dict:
        #         value = losses_dict[key]
        #         total_loss[key] += value.item()
        # all_dummy_idx = dummy_dict['dummy_idx'].all(dim=1) #(bs) #num_queries개수만큼 더미가 모두 뽑혔을 때 dummy_idx는 True
        # num_all_dummy = all_dummy_idx.sum()
        # num_not_all_dummy += (batch_sample-num_all_dummy)

            
        batch_sample_size = [batch_sample, num_no_target, batch_sample-num_no_target]
        if "loss_exis_score" in losses_dict:
            exis_enc_flag = True
            loss_exis_score = losses_dict.pop("loss_exis_score")
            for key, size in zip(exis_keys, batch_sample_size):
                if size > 0:
                    value = loss_exis_score[key]
                    exis_total_loss[key] += value.item() * size
            loss_exis_score_mean = loss_exis_score.get("loss_score_mean", torch.tensor([0.0], device=device))
            if loss_weight_flag == True:
                loss_weight['det_loss'], loss_weight['exis_loss'] = 1.0, 1.0 #0.2, 0.8
        else:
            exis_enc_flag = False
            loss_exis_score_mean = torch.tensor([0.0], device=device)
            
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
            min_num_f1_0_ot_sample_ratio = min_num_f1_0_ot_sample/num_ot if num_ot>0 else float('nan')
            

        if exis_enc_flag:
            nt_topk_sum = topk_per_batch_mean[no_target].sum()
            nt_topk_sum_all += nt_topk_sum
            ot_topk_sum = topk_per_batch_mean[ot_bool].sum()
            ot_topk_sum_all += ot_topk_sum
            mt_topk_sum = topk_per_batch_mean[mt_bool].sum()
            mt_topk_sum_all += mt_topk_sum

            if topk_cos_sim_loss_flag:
                gt_topk_mean = ~no_target
                gt_topk_mean = gt_topk_mean.float()
                topk_cos_sim_loss = F.mse_loss(topk_per_batch_mean, gt_topk_mean, reduction="mean")
            else: topk_cos_sim_loss = 0

        dummy_token_diversity_loss = losses_dict.pop("dummy_token_diversity_loss", torch.tensor([0.0], device=device))
        nt_dummy_loss = losses_dict.pop("nt_dummy_loss", torch.tensor([0.0], device=device))
        others_dummy_loss = losses_dict.pop("others_dummy_loss", torch.tensor([0.0], device=device))
        loss_det = losses_dict.get("loss_total", torch.tensor([0.0], device=device))+losses_dict.get("loss_det", torch.tensor([0.0], device=device))
        loss_mask = losses_dict.pop("loss_mask", torch.tensor([0.0], device=device))
        loss = loss_det * loss_weight['det_loss'] + loss_mask + loss_exis_score_mean * loss_weight['exis_loss'] + topk_cos_sim_loss#+ dummy_token_diversity_loss #+ nt_dummy_loss + others_dummy_loss
        optimizer.zero_grad()
        if cfg.use_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
            
        # for name, param in model.named_parameters():
        #     if 'head' in name:
        #         if param.requires_grad:
        #             if param.grad is not None:
        #                 print(f"{name}: grad exists ✅, grad norm = {param.grad.norm().item():.6f}")
        #             else:
        #                 print(f"{name}: grad is None ❌")
                        
        if cfg.grad_norm_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip)
        optimizer.step()

        if cfg.ema:
            model_ema.update_params()

            
            
        # if cfg.distributed:
        #     loss_det = reduce_mean(loss_det)
        #     loss_mask = reduce_mean(loss_mask)

        if not isinstance(predictions, list):
            predictions_list = [predictions]
        else:
            predictions_list = predictions
            
        # statistics loss
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
                    batch_det_acc, _, _ = accuracy(
                        pred_bboxes,
                        gt_bbox,
                        pred_masks,
                        gt_mask,
                        is_crowd=is_crowd,
                        device=device,
                    )
                    if cfg.distributed:
                        batch_det_acc = reduce_mean(batch_det_acc)
                det_acc_list[predict_type].append(batch_det_acc.item())
                # loss_det_list[predict_type].append(loss_det.item())
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
                # loss_det_list[predict_type].append(loss_det.item())
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
        if is_main() and ((batch + 1) % cfg.log_interval == 0 or batch + 1 == num_batches):
            loss_str_list = ["{}:{:.3f}".format(loss_n.split("loss_")[-1], sum(loss_v)/len(loss_v)) for loss_n, loss_v in loss_det_list.items()]
            loss_str =  "loss:["+" ".join(loss_str_list) +"]"
            logger = get_root_logger()
            if not cfg["dataset"] == "GRefCOCO":
                ACC_str_list = ["{}Acc:{:.2f}, ".format(map_dict[i], det_acc_dict[map_dict[i]]) for i in range(len(predictions_list))]
                ACC_str = "".join(ACC_str_list)
                logger.info(
                    f"train-epoch[{epoch+1}]-[{batch+1}/{num_batches}] "
                    + f"time:{(time.time()- end):.2f}, data_time: {data_time:.2f}, "
                    # + f"loss_det:{sum(loss_det_list[predict_type]) / len(loss_det_list[predict_type]) :.4f}, "
                    + f"{loss_str}, "
                    + f"lr:{optimizer.param_groups[0]['lr']:.6f}, "
                    # + f"DetACC@0.5: {det_acc:.2f}, "
                    + ACC_str
                )
            else:
                F1_Score_str_list = [
                    "{}_f1: {:.2f}, ".format(map_dict[i], f1_score_acc_dict[map_dict[i]]) for i in range(len(predictions_list))
                ]
                F1_Score_str = "".join(F1_Score_str_list)
                n_acc_str_list = ["{}_Nacc: {:.2f}, ".format(map_dict[i], n_acc_dict[map_dict[i]]) for i in range(len(predictions_list))]
                n_acc_str = "".join(n_acc_str_list)
                ACC_str_list = ["{}Acc:{:.2f}, ".format(map_dict[i], det_acc_dict[map_dict[i]]) for i in range(len(predictions_list))]
                ACC_str = "".join(ACC_str_list)
                logger.info(
                    f"train-epoch[{epoch+1}]-[{batch+1}/{num_batches}] "
                    + f"time:{(time.time()- end):.2f}, data_time: {data_time:.2f}, "
                    # + f"loss_det:{sum(loss_det_list[predict_type]) / len(loss_det_list[predict_type]) :.4f}, "
                    # +f"{loss_str}, "
                    + f"lr:{optimizer.param_groups[0]['lr']:.6f}, "
                    + F1_Score_str
                    + n_acc_str
                    + ACC_str
                    #+ f"num_acc_dummy: {dummy_dict['num_accurate_dummy'].item()}"
                )
                #Tensorboard
                x_step = epoch*num_batches + batch + 1

                for loss_n, loss_v in loss_det_list.items():
                    loss_n = f"{loss_n.split('loss_')[-1]}"
                    avg_loss = sum(loss_v)/len(loss_v)
                    avg_loss_dict_[loss_n] = avg_loss
                writer.add_scalars(f"Loss/train", avg_loss_dict_, x_step)
                    
                # #전체 train detection loss
                # if ((batch+1)-num_not_valid_loss) !=0:
                #     avg_loss_dict = {k: v / ((batch+1)-num_not_valid_loss) for k, v in total_loss.items()}
                #     #train loss
                #     writer.add_scalars(f"Loss/train", avg_loss_dict, x_step) 
                #train F1, N-acc
                if batch + 1 == num_batches:
                    print("more_than_ten_target", more_than_ten_target)
                    #전체 N-acc
                    N_acc_all = nt_all["TP"] / (nt_all["TP"] + nt_all["FN"]) if nt_all["TP"] != 0 else torch.tensor(0.0, device=device)
                    N_acc_all = N_acc_all.float() * 100
                    #전체 F1
                    F1_score_all = correct_image_all / num_image_all
                    F1_score_all = F1_score_all.float() * 100
                    writer.add_scalars(f"metric/f1", {"train_f1":F1_score_all.item()}, x_step)
                    writer.add_scalars(f"metric/N-acc", {"train_N-acc":N_acc_all.item()}, x_step)
                    #전체 det acc
                    det_acc_all = num_correct_ot_all/num_ot_all
                    det_acc_all = det_acc_all.float()*100
                    writer.add_scalars(f"metric/det_acc", {"train_det_acc":det_acc_all.item()}, x_step)
                    if exis_enc_flag :
                        #전체 exis loss
                        sample_sizes = [total_sample, num_no_target_all, total_sample-num_no_target_all]
                        avg_exis_loss_dict = {k:v / sample_size for (k, v), sample_size in zip(exis_total_loss.items(), sample_sizes)}
                        writer.add_scalars(f"Exis_Loss/train", avg_exis_loss_dict, x_step)
                        print(sample_sizes)

                        Exis_distinguish = {"nt": nt_topk_sum_all/num_no_target_all, "ot": ot_topk_sum_all/num_ot_all, "mt": mt_topk_sum_all/num_mt_all}
                        
                        writer.add_scalars(f"Exis_distinguish/train", Exis_distinguish, x_step)
                        if topk_cos_sim_loss_flag:
                            writer.add_scalars(f"topk_cos_sim_loss_flag", {"train":topk_cos_sim_loss.item()}, x_step)
                    
                    if dummy_dict is not None:
                        #해당 배치의 diversity loss
                        writer.add_scalar(f"dummy_diversity_loss/train", dummy_token_diversity_loss, x_step)
                        #해당 배치의 dummy enhance loss
                        writer.add_scalar(f"dummy_enhance_loss/train/nt", nt_dummy_loss, x_step)
                        writer.add_scalar(f"dummy_enhance_loss/train/others", others_dummy_loss, x_step)
                        #전체 Dummy precision
                        dummy_precision_all = dummy_dict_all['num_accurate_dummy']/dummy_dict_all['num_all_dummy']
                        writer.add_scalars(f"dummy_metric/train", {"dummy_precision":dummy_precision_all.item()}, x_step)
                        #전체 Dummy recall
                        dummy_recall_all = dummy_dict_all['num_accurate_dummy']/num_no_target_all
                        writer.add_scalars(f"dummy_metric/train", {"dummy_recall":dummy_recall_all.item()}, x_step)
                        dummy_f1_all = 2*(dummy_precision_all*dummy_recall_all)/(dummy_precision_all+dummy_recall_all)
                        writer.add_scalars(f"dummy_metric/train", {"dummy_f1":dummy_f1_all.item()}, x_step)
                        writer.add_scalars(f"dummy_ratio/train", {"dummy_num/total_size":dummy_dict_all['num_all_dummy'].item()/sample_sizes[0]}, x_step)

                        #min_num_f1_0_sample_ratio
                        writer.add_scalars(f"min_num_f1_0_sample_ratio/train", {"min_num_f1_0_mt_sample_ratio":min_num_f1_0_mt_sample_ratio, "min_num_f1_0_ot_sample_ratio": min_num_f1_0_ot_sample_ratio}, x_step)
                        
                        nt_denom = dummy_dict_all['sum_part_dummy_of_nt'].item()
                        others_denom = dummy_dict_all['sum_part_dummy_of_others'].item()

                        if nt_denom > 1e-6 and others_denom > 1e-6:
                            #part dummy ratio
                            writer.add_scalars(f"extract_part_dummy/ratio/train", {"no-target":dummy_dict_all['sum_dummy_ratio_of_part_dum_nt'].item()/nt_denom, "others":dummy_dict_all['sum_dummy_ratio_of_part_dum_others'].item()/others_denom}, x_step)
                        #part dummy 수
                        writer.add_scalars(f"extract_part_dummy/num_sample/train", {"no-target": dummy_dict_all['sum_part_dummy_of_nt'].item(), "others":dummy_dict_all['sum_part_dummy_of_others'].item()}, x_step)
                        
                        #print('dummy_precision_all', dummy_precision_all.item(), 'dummy_recall_all', dummy_recall_all.item(), 'dummy_f1_all', dummy_f1_all.item())
                        #dev
                        if dev is not None:
                            writer.add_scalars(f"dev/train/no_target", {
                                'sim':  dev['no_target']['sim']/num_no_target_all,
                                'scaled':  dev['no_target']['scaled_sim']/num_no_target_all
                            }, x_step)
                            writer.add_scalars(f"dev/train/others", {
                                'sim':  dev['others']['sim']/ (total_sample - num_no_target_all),
                                'scaled':  dev['others']['scaled_sim']/ (total_sample - num_no_target_all)
                            }, x_step)
                        #ratio_under/over_cross_blah
                        writer.add_scalars(f"ratio_cross_blah/train", {
                            'yes_dum_no_target_over_cross' : dummy_dict_all['ratio_over_cross_blah']/batches_for_check_over,
                            'yes_dum_no_target_under_cross' : dummy_dict_all['ratio_under_cross_blah']/batches_for_check_under
                        }, x_step)

                
                    #print('num_more_than_two_target', more_than_two_target)
        end = time.time()
