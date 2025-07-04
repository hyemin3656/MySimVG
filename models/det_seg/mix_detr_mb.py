import torch
import numpy as np
import numpy
from simvg.models import MODELS
from mmdet.core import BitmapMasks
import pycocotools.mask as maskUtils
from .one_stage import OneStageModel
from detectron2.modeling import detector_postprocess
import torch.nn.functional as F
from detrex.layers.box_ops import box_cxcywh_to_xyxy
from detectron2.structures import Boxes, ImageList, Instances
from simvg.models.compute_score import ExisEcoder
from simvg.models.vis_encs.beit.modeling_utils import _get_base_config, _get_large_config
import torch.nn as nn
import datetime
import matplotlib.pyplot as plt
import os
from torchvision.ops import sigmoid_focal_loss

@MODELS.register_module()
class MIXDETRMB(OneStageModel):
    def __init__(self, word_emb, num_token, vis_enc, lan_enc, head, fusion):
        super(MIXDETRMB, self).__init__(word_emb, num_token, vis_enc, lan_enc, head, fusion)
        self.patch_size = vis_enc["patch_size"]
        
        self.beit_sentence_token_flag=False ####
        self.exis_enc_sentence_token_flag=True
        self.exis_encoder_flag=True
        
        self.exis_proj = nn.Linear(768, 1) 
        self.now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dev = {'no_target': {'dotpro': 0, 'scaled': 0 }, 'others': {'dotpro': 0, 'scaled': 0 }}

        #---ExisEncoder---
        if self.exis_encoder_flag==True:
            #beit3의 embed_dim 가져오기
            vit_type = vis_enc["vit_type"]
            if vit_type == "base":
                args = _get_base_config()
            elif vit_type == "large":
                args = _get_large_config()
            else:
                raise TypeError("please select the <vit_type> from ['base','large']")
            embed_dim = args.encoder_embed_dim
            self.exisenc = ExisEcoder(embed_dim, self.exis_enc_sentence_token_flag)

    def forward_train(
        self,
        img,
        ref_expr_inds,
        img_metas,
        train_flag=True,
        text_attention_mask=None,
        gt_bbox=None,
        gt_mask_vertices=None,
        rescale=False,
        epoch=None,
        batch=None,
        batches=None
    ):
        """Args:
        img (tensor): [batch_size, c, h_batch, w_batch].

        ref_expr_inds (tensor): [batch_size, max_token].

        img_metas (list[dict]): list of image info dict where each dict
            has: 'img_shape', 'scale_factor', and may also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys see
            `seqtr/datasets/pipelines/formatting.py:CollectData`.

        gt_bbox (list[tensor]): [4, ], in [tl_x, tl_y, br_x, br_y] format,
            the coordinates are in 'img_shape' scale.

        gt_mask_vertices (list[tensor]): [batch_size, 2, num_ray], padded values are -1,
            the coordinates are in 'pad_shape' scale.

        rescale (bool): whether to rescale predictions from `img_shape`/`pad_shape`
            back to `ori_shape`.

        """
        B, _, H, W = img.shape
        # print(ref_expr_inds) 
        # print(img_metas) #category_id #expression
        #print('ref_expr_inds', ref_expr_inds)

        img_feat, text_feat, cls_feat = self.extract_visual_language(img, ref_expr_inds, text_attention_mask, sentence_token_flag=self.beit_sentence_token_flag)
        
        if self.beit_sentence_token_flag==True or self.exis_encoder_flag==True:
            
            if self.beit_sentence_token_flag==True:
                text_feat, sent_feat = text_feat[:, :-1, :], text_feat[:, -1, :]
            else:
                #mask_for_exisenc
                spec_token_mask = (ref_expr_inds == 2) | (ref_expr_inds == 0)
                combined_mask = spec_token_mask | text_attention_mask
                sent_feat = self.exisenc(img_feat, text_feat, text_mask=combined_mask, img_metas=img_metas) #(bs, embed_dim)
            
            #FC -> exis score
            exis_scores = self.exis_proj(sent_feat) #(bs, 1)
            exis_probs = torch.sigmoid(exis_scores) #(bs, 1)

            #loss
            gt_scores = torch.tensor([0 if img_meta["target"][0]["category_id"] == -1 else 1 for img_meta in img_metas], device=exis_probs.device).float() #(bs)
            gt_scores_bool = gt_scores.bool()
            #For BCE
            # weights = torch.ones_like(gt_scores)
            # weights[gt_scores == 0] = 8.0
            
            # loss 계산 - focal loss로 변경
            per_sample_los_focal = sigmoid_focal_loss(
                inputs=exis_scores.squeeze(-1), #positive class에 대한 logit
                targets=gt_scores,
                alpha=0.25,
                gamma=2.0,
                reduction='none'
            )
            
            
            # per_sample_los = F.binary_cross_entropy(
            #     exis_probs.squeeze(-1),
            #     gt_scores,
            #     weight=weights,
            #     reduction='none'
            # )
            
            loss_score = per_sample_los_focal.mean()
            
            #loss왜곡 때문에 BCE로 시각화
            per_sample_los_bce = F.binary_cross_entropy(
                exis_probs.squeeze(-1), #positive class에 대한 prob
                gt_scores,
                reduction='none'
            )
            #no-target loss
            no_target_per_sample_los = per_sample_los_bce[~gt_scores_bool] #weight BCE 시 나누기 필요
            no_target_los_mean = no_target_per_sample_los.mean()
            #others
            others_per_sample_los = per_sample_los_bce[gt_scores_bool]
            others_los_mean = others_per_sample_los.mean()

            #for LGQS
            scores = exis_probs.expand(-1, text_feat.size(1)) #(bs, max_seq_len)

            #box plot---------------
            if (epoch + 1 in [0, 1] or (epoch + 1) % 5 == 0) and (batch + 1 == batches): #epoch 0일때 train, epoch 5의 배수일 때 train, validation 저장
                others = exis_probs[gt_scores_bool, :] #others에 해당하는 샘플들의 exis_prob 리스트
                no_target = exis_probs[~gt_scores_bool, :] #no_target에 해당하는 샘플들의 exis_prob 리스트
                plt.boxplot([others.squeeze(-1).detach().cpu().numpy(), no_target.squeeze(-1).detach().cpu().numpy()], labels=['others', 'no_target'])
                plt.title(f"existence score boxplot_ {epoch+1}epoch")
                plt.grid(True)
                plt.ylim(0, 1)

                train_save_dir = f"{self.now}/box_plot/train"
                val_save_dir = f"{self.now}/box_plot/val"
                os.makedirs(train_save_dir, exist_ok=True)
                os.makedirs(val_save_dir, exist_ok=True)
                
                if train_flag==True:
                    filename = os.path.join(train_save_dir, f"epoch_{epoch}.png")
                else:
                    filename = os.path.join(val_save_dir, f"epoch_{epoch}.png")
                plt.savefig(filename)
                plt.close()  
        else:
            scores = None
        #print(text_feat)
        #img_feat:(bs, num_patches, embed_dim)
        #text_feat:(bs, max_seq_len, embed_dim)
        #cls_feat : (bs, embed_dim)
        #text_attention_mask:(bs, max_seq_len)
        
        img_feat_trans = img_feat.transpose(-1, -2).reshape(B, -1, H // self.patch_size, W // self.patch_size)

        print(text_attention_mask[:, 0])
        print(text_attention_mask[:, -1])
        text_attention_mask[:, 0], text_attention_mask[:, -1] = 1, 1

        #detr head
        losses_dict, output, dummy_dict, similarity, scaled_similarity, scale_factor = self.head.forward_train(
            img_feat_trans, img_feat, img_metas, text_feat=text_feat, text_scores=scores, text_mask=text_attention_mask, gt_bbox=gt_bbox, epoch=epoch #img_feat, img_metas, cls_feat=cls_feat, gt_bbox=gt_bbox, text_feat=text_feat, text_mask=text_attention_mask
        )
        #output_token_branch = output["token_branch_output"]
        #output_decoder_branch = output["decoder_branch_output"]

        
        
        output_decoder_branch = output

        with torch.no_grad():
            if img_metas[0].get("target", None) is None:
                #predictions_token_branch = self.get_predictions(output_token_branch, img_metas, rescale=rescale)
                predictions_decoder_branch = self.get_predictions(output_decoder_branch, img_metas, rescale=rescale)
            else: # grec output
                #predictions_token_branch = self.get_predictions_grec(output_token_branch, img_metas, rescale=rescale)
                predictions_decoder_branch = self.get_predictions_grec(output_decoder_branch, img_metas, dummy_dict, rescale=rescale)
            
        predictions = [predictions_decoder_branch]

        #score loss 추가
        losses_dict['loss_exis_score'] = {'loss_score_mean' : loss_score, 'no_target_los_mean' : no_target_los_mean, 'others_los_mean' : others_los_mean}
        
        #시각화--------------------------------------------------------
        if (epoch + 1 in [0, 1] or (epoch + 1) % 5 == 0): 
            #similarity 마스킹
            # 1) bool mask로 변환 (필요하다면)
            attn_mask = text_attention_mask.bool()           # (bs, max_seq_len)
            
            # 2) 패치 축 추가 및 확장
            attn_mask = attn_mask.unsqueeze(1)               # (bs, 1, max_seq_len)
            attn_mask = attn_mask.expand(-1, similarity.size(1), -1)  
            # → (bs, num_patches+1, max_seq_len)
            
            # 3) -∞ 로 마스킹
            similarity = similarity.masked_fill(attn_mask, float('-inf'))
            #편차 계산
            dotpro_np = similarity.detach().cpu().numpy() #(batch_size, num_patches+1, max_seq_len)
            scaled_np = scaled_similarity.detach().cpu().numpy() #(batch_size, num_patches+1, max_seq_len)
            gt_bool = gt_scores.bool().detach().cpu().numpy()
            batch_size = dotpro_np.shape[0]
            
            
            max_dotpro_np = np.max(dotpro_np, axis = -1) #(batch_size, num_patches+1)
            devi_dotpro = np.max(max_dotpro_np) - max_dotpro_np[:, -1] #(bs)
            max_scaled_np = np.max(scaled_np, axis = -1)
            devi_scaled = np.max(max_scaled_np) - max_scaled_np[:, -1] #(bs)

            others_devi_dotpro = devi_dotpro[gt_bool]
            no_target_devi_dotpro = devi_dotpro[~gt_bool]

            others_devi_scaled = devi_scaled[gt_bool]
            no_target_devi_scaled = devi_scaled[~gt_bool]
            
            self.dev['no_target']['dotpro'] += no_target_devi_dotpro.sum()
            self.dev['no_target']['scaled'] += no_target_devi_scaled.sum()
            self.dev['others']['dotpro'] += others_devi_dotpro.sum()
            self.dev['others']['scaled'] += others_devi_scaled.sum()
            
            dev = self.dev
            
            if (batch + 1 == batches):
                self.dev = {'no_target': {'dotpro': 0, 'scaled': 0 }, 'others': {'dotpro': 0, 'scaled': 0 }}
                #히트맵 시각화
                if train_flag == True:
                    output_base = f"{self.now}/heatmap/train/epoch_{epoch}"
                else:
                    output_base = f"{self.now}/heatmap/val/epoch_{epoch}"
                #(batch_size, num_patches+1, max_seq_len)인 similarity와 scaled_similarity
                    
                for i in range(batch_size):
                    # 샘플별 group 결정
                    group = "others" if gt_bool[i] else "no_target"
                    #샘플별 exis prob
                    exis_prob_per_sample = exis_probs[i].item()
                    #샘플별 더미 추출 여부
                    dummy_extract = dummy_dict['dummy_idx'][i].item()
                    if dummy_extract ==True:
                        dummy_extract='Yes'
                    else: dummy_extract='No'
                    
                
                    # 그룹 폴더(others/, no_target/)만 생성
                    group_folder = os.path.join(output_base, group)
                    os.makedirs(group_folder, exist_ok=True)

                    # 해당 샘플의 히트맵 데이터
                    m = dotpro_np[i]   # (num_patches+1, max_seq_len)
                    s = scaled_np[i]   # same shape
                    
                    valid_cols = ~np.any(np.isneginf(m), axis=0)
                    
                    m = m[:, valid_cols]
                    s = s[:, valid_cols]

                    # 두 배열의 전체 범위 계산
                    vmin = min(m.min(), s.min())
                    vmax = max(m.max(), s.max())
                    
                    # 히트맵 나란히 그리기
                    fig, axes = plt.subplots(1, 2, figsize=(11, 12), tight_layout=True)
                    im0 = axes[0].imshow(m, aspect='auto', vmin=vmin, vmax=vmax, interpolation='nearest')
                    axes[0].set_title(f"{group} sample {i} – Dotproduct Sim")
                    fig.colorbar(im0, ax=axes[0])
                
                    im1 = axes[1].imshow(s, aspect='auto', vmin=vmin, vmax=vmax, interpolation='nearest')
                    axes[1].set_title(f"{group} sample {i} – Scaled Sim")
                    fig.colorbar(im1, ax=axes[1])
                
                    # axis=1 방향 최대값 위치에 빨간 × 표시
                    for ax, arr in zip(axes, (m, s)):
                        row_max_cols = np.argmax(arr, axis=1)
                        rows = np.arange(arr.shape[0])
                        ax.scatter(row_max_cols, rows, marker='x', c='red', s=20)
                        ax.set_xlabel("token index")
                        ax.set_ylabel("patch index")
                        
                    #scale factor
                    scale_not_dum = scale_factor[i, 0, 0]
                    scale_dum = scale_factor[i, -1, 0]
                
                    # 파일명에 샘플 인덱스 포함해서 저장
                    save_path = os.path.join(group_folder, f"{group}sample_{i}.png")
                    fig.text(0.9, 0.97, f"exis prob: {exis_prob_per_sample:.2f}\nSF : {scale_not_dum:.2f}| {scale_dum:.2f}\ndummy extract?: {dummy_extract}")
                    fig.savefig(save_path, dpi=130, bbox_inches='tight')
                    plt.close(fig)
        else:
            dev = None
        return losses_dict, predictions, dummy_dict, dev

    def extract_visual_language(self, img, ref_expr_inds, text_attention_mask=None, sentence_token_flag=False):
        x, y, c= self.vis_enc(img, ref_expr_inds, text_attention_mask, sentence_token_flag=sentence_token_flag)
        return x, y, c

    @torch.no_grad()
    def forward_test(
        self,
        img,
        ref_expr_inds,
        img_metas,
        text_attention_mask=None,
        with_bbox=False,
        with_mask=False,
        rescale=False,
    ):
        """Args:
        img (tensor): [batch_size, c, h_batch, w_batch].

        ref_expr_inds (tensor): [batch_size, max_token], padded value is 0.

        img_metas (list[dict]): list of image info dict where each dict
            has: 'img_shape', 'scale_factor', and may also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys see
            `rec/datasets/pipelines/formatting.py:CollectData`.

        with_bbox/with_mask: whether to generate bbox coordinates or mask contour vertices,
            which has slight differences.

        rescale (bool): whether to rescale predictions from `img_shape`/`pad_shape`
            back to `ori_shape`.
        """

        B, _, H, W = img.shape
        img_feat, text_feat, cls_feat = self.extract_visual_language(img, ref_expr_inds, text_attention_mask)
        loss_score, scores = self.exisenc(img_feat, text_feat, text_mask=text_attention_mask, img_metas=img_metas)
        img_feat_trans = img_feat.transpose(-1, -2).reshape(B, -1, H // self.patch_size, W // self.patch_size)
        output = self.head.forward_test(img_feat_trans, img_feat, img_metas, text_feat=text_feat, text_scores=scores, text_mask=text_attention_mask, with_bbox=with_bbox, with_mask=with_mask) #cls_feat = cls_feat, text_mask=text_attention_mask

        #output_token_branch = output["token_branch_output"]
        output_decoder_branch = output

        with torch.no_grad():
            if img_metas[0].get("target", None) is None:
                #predictions_token_branch = self.get_predictions(output_token_branch, img_metas, rescale=rescale)
                predictions_decoder_branch = self.get_predictions(output_decoder_branch, img_metas, rescale=rescale)
            else: # grec output
                #predictions_token_branch = self.get_predictions_grec(output_token_branch, img_metas, rescale=rescale)
                predictions_decoder_branch = self.get_predictions_grec(output_decoder_branch, img_metas, rescale=rescale)
            
        predictions = [predictions_decoder_branch] #predictions_token_branch

        return predictions

    def get_predictions(self, output, img_metas, rescale=False):
        box_cls = output["pred_logits"]
        box_pred = output["pred_boxes"]
        image_sizes = [img_meta["img_shape"] for img_meta in img_metas]
        if box_cls is None:
            return dict(pred_bboxes=None, pred_masks=None, predict_classes=None)
        results = self.head.inference(box_cls, box_pred, image_sizes)
        # processed_results = []
        pred_bboxes = []
        predict_classes = []
        for results_per_image, img_meta in zip(results, img_metas):
            image_size = img_meta["img_shape"]
            height = image_size[0]
            width = image_size[1]
            r = detector_postprocess(results_per_image, height, width)
            # infomation extract
            pred_boxes = r.pred_boxes
            scores = r.scores
            pred_class = r.pred_classes
            # best index
            best_ind = torch.argmax(scores)
            pred_box = pred_boxes[int(best_ind)].tensor
            if rescale:
                scale_factors = img_meta["scale_factor"]
                pred_box /= pred_box.new_tensor(scale_factors)
            predict_classes.append(pred_class)
            pred_bboxes.append(pred_box)
            # processed_results.append({"instances": r})
            
        pred_bboxes = torch.cat(pred_bboxes, dim=0)
        predict_classes = torch.cat(predict_classes, dim=0)
        pred_masks = None
        return dict(pred_bboxes=pred_bboxes, pred_masks=pred_masks, predict_classes=predict_classes)
    
    def get_predictions_grec(self, output, img_metas, dummy_dict, rescale=False):
        #모델의 출력 결과로부터 최종 예측 결과(바운딩 박스, 클래스 등)를 추출
        box_cls = output["pred_logits"] # [bs, num_query, 2]
        box_pred = output["pred_boxes"] # [bs, num_query, 4]
        image_sizes = [img_meta["img_shape"] for img_meta in img_metas]
        if box_cls is None:
            return dict(pred_bboxes=None, pred_masks=None, predict_classes=None)
        results = self.head.inference(box_cls, box_pred, image_sizes) 
        # processed_results = []
        pred_bboxes = []
        #결과를 이미지별로 순회
        for results_per_image, img_meta, is_dummy in zip(results, img_metas, dummy_dict['dummy_idx']):
            image_size = img_meta["img_shape"]
            height = image_size[0]
            width = image_size[1]

            if is_dummy:
                new_box = torch.tensor([[0.0, 0.0, 1e-6, 1e-6]], device=results_per_image.pred_boxes.tensor.device)
                results_per_image.pred_boxes = Boxes(new_box)
            #예측결과 후처리 - 이미지 크기에 맞는 최종 결과 생성 (ex. 좌표 변환, 스케일 복원)
            r = detector_postprocess(results_per_image, height, width)
            if is_dummy:
                new_box = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=results_per_image.pred_boxes.tensor.device)
                r.pred_boxes = Boxes(new_box)

            # infomation extract
            pred_box = r.pred_boxes.tensor #(n_q, 4)
            score = r.scores #(n_q)
            pred_class = r.pred_classes #(n_q)
            if rescale:
                scale_factors = img_meta["scale_factor"]
                pred_box /= pred_box.new_tensor(scale_factors)
            cur_predict_dict = {
                "boxes":pred_box,
                "scores":score,
                "labels":pred_class
            }
            pred_bboxes.append(cur_predict_dict)
            # processed_results.append({"instances": r})
        pred_masks = None
        return dict(pred_bboxes=pred_bboxes, pred_masks=pred_masks)