from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sigmoid

from detrex.layers.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.layers.mlp import MLP
from detrex.modeling.criterion.criterion import SetCriterion
from detrex.layers.position_embedding import PositionEmbeddingLearned, PositionEmbeddingSine
from detrex.modeling.matcher.matcher import HungarianMatcher
from detectron2.structures import Boxes, ImageList, Instances

from .transformer import DetrTransformer, DetrTransformerEncoder, DetrTransformerDecoder
from simvg.models import HEADS

#Token branch 사용 X
@HEADS.register_module()
class DETRHead(nn.Module):
    def __init__(
        self,
        num_queries=100,
        in_channels=768, #beit3 output feature의 embed_dim
        embed_dim=256,
        num_classes=1, #config에도 1
        aux_loss=True,
        num_encoder_layers=6,
        num_decoder_layers=6,
        only_decoder=False,
        language_guided_query_selection_flag = True
    ):
        super(DETRHead, self).__init__()
        self.num_queries = num_queries
        self.transformer = DetrTransformer(
            encoder=DetrTransformerEncoder(
                embed_dim=embed_dim,
                num_heads=8,
                attn_dropout=0.1,
                feedforward_dim=2048,
                ffn_dropout=0.1,
                num_layers=num_encoder_layers,
                post_norm=False,
            ),
            decoder=DetrTransformerDecoder(
                embed_dim=embed_dim,
                num_heads=8,
                attn_dropout=0.1,
                feedforward_dim=2048,
                ffn_dropout=0.1,
                num_layers=num_decoder_layers,
                return_intermediate=True,
                post_norm=True,
            ),
            only_decoder=only_decoder,
        )
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1) #image feature projection
        self.query_embed = nn.Embedding(num_queries, embed_dim) #lgqs X 시 쿼리 초기화
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=embed_dim // 2,
            temperature=10000,
            normalize=True,
        )
        self.language_guided_query_selection_flag = language_guided_query_selection_flag
        if self.language_guided_query_selection_flag:
            self.dummy_token = nn.Parameter(torch.randn(1, num_queries, in_channels)) 
            self.query_proj = nn.Linear(in_channels, embed_dim)

        # define classification head and box head
        self.class_embed = nn.Linear(embed_dim, num_classes + 1) 
        self.bbox_embed = MLP(input_dim=embed_dim, hidden_dim=embed_dim, output_dim=4, num_layers=3)

        matcher = HungarianMatcher(
            cost_class=1,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="ce_cost",
        )
        self.criterion = SetCriterion(
            num_classes=num_classes, #1
            matcher=matcher,
            weight_dict={
                "loss_class": 1,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
            },
            loss_class_type="ce_loss", #ce loss or focal loss
            eos_coef=0.1,
        )

        if self.aux_loss:
            weight_dict = self.criterion.weight_dict
            aux_weight_dict = {}
            for i in range(self.transformer.decoder.num_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
            self.criterion.weight_dict = weight_dict

    def prepare_targets(self, targets, img_metas):
        new_targets = []
        for target_bbox, img_meta in zip(targets, img_metas): #targets와 img_metas는 현재 미니배치 전체 샘플 담고 있음
            h, w = img_meta["img_shape"][:2] #resized image shape (640, 640)
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=target_bbox.device)
            if len(target_bbox.shape) == 1: 
                target_bbox = target_bbox.unsqueeze(0)
                gt_classes = torch.zeros(1, device=target_bbox.device).long()
            else:  # for grec # TODO None object can be set as label 1 ? or just set no GT #single target / no-target([0, 0, 0, 0])도 dummy차원 존재해서 이곳에해당
                assert int(target_bbox.shape[0]) == len(img_meta["target"])
                #Non-object -> 1/others -> 0
                gt_classes = torch.tensor([1 if t["category_id"] == -1 else 0 for t in img_meta["target"]], device=target_bbox.device).long()
            gt_boxes = target_bbox.float() / image_size_xyxy #[0, 1] 정규화
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes) #(center_x, center_y, width, height) 형태로 변환
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes}) 
            #lables 예시 ex.대응되는 expression이 이미지내에 없는 경우 -> [1] / 대응되는 expression이 2개의 박스일 경우 -> [0, 0]
        return new_targets
    
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # 중간레이어 값들 저장
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])] 

    def x_mask_pos_enc(self, x, img_metas):
        batch_size = x.size(0)
        input_img_h, input_img_w = img_metas[0]["batch_input_shape"]
        x_mask = x.new_ones((batch_size, input_img_h, input_img_w))
        # CAUTION: do not support random flipping
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]["img_shape"]
            x_mask[img_id, :img_h, :img_w] = 0

        x_mask = F.interpolate(x_mask.unsqueeze(1), size=x.size()[-2:]).to(torch.bool).squeeze(1)

        x_pos_embeds = self.position_embedding(x_mask)

        # x_mask = x_mask.view(batch_size, -1)
        # x_pos_embeds = x_pos_embeds.view(
        #     batch_size, self.d_model, -1).transpose(1, 2)

        return x_mask, x_pos_embeds

    def language_guided_query_selection(self, text_feat, img_feat, text_scores, text_mask, epoch=None):
        #text_scores : (bs, 1)
        nan_mask = text_mask.bool().unsqueeze(1) #(bs, 1, max_seq_len)
        dummy_token_expanded = self.dummy_token.expand(img_feat.size(0), -1, -1)
        # if epoch>=10:
        #     img_feat_with_dummy = torch.cat([img_feat, dummy_token_expanded], dim=1) #(bs, num_patches + num_query, embed_dim)
        # else:
        #     img_feat_with_dummy = img_feat

        img_feat_with_dummy = torch.cat([img_feat, dummy_token_expanded], dim=1) #(bs, num_patches + num_query, embed_dim)
        # --- reshape for broadcasting ---
        img_feat_exp = img_feat_with_dummy.unsqueeze(2)  # (bs, N_p + num_query, 1, dim)
        text_feat_exp = text_feat.unsqueeze(1)           # (bs, 1, max_seq_len, dim)
    
        # --- cosine similarity 계산 ---
        similarity_cos = F.cosine_similarity(img_feat_exp, text_feat_exp, dim=-1)  # (bs, N_p + num_query, max_seq_len)
        #--- cosine similarity + scaling ---
        similarity_norm = (similarity_cos + 1) / 2 
        #text feature와 image feature 내적
        #similarity_dotpro = torch.matmul(img_feat_with_dummy, text_feat.transpose(-1, -2)) #(batch_size, num_patches+1, max_seq_len)

        #B, N, T = similarity_dotpro.shape
        #sim_flat = similarity_dotpro.view(B, -1)
        
        # # 각 샘플 별 min, max 계산: shape (B, 1)
        #min_vals = sim_flat.min(dim=1)[0].view(B, 1, 1)
        #max_vals = sim_flat.max(dim=1)[0].view(B, 1, 1)
        
        # # 정규화
        #similarity_dotpro_norm = (similarity_dotpro - min_vals) / (max_vals - min_vals)
        #score로 내적값 조절
        #inver_sim =  2**(2 - text_scores) - 2 #(batch_size, max_seq_len)
        #inver_sim = inver_sim.unsqueeze(1).expand(-1, img_feat_with_dummy.size(1), -1)
        #base = (1 - torch.sigmoid(similarity_dotpro_norm)) + 1e-6 #.clamp(min=1e-6) #gradient NaN 방지
        #inver_sim = inver_sim.clamp(min=0, max=2)
        #scale_factor = base ** inver_sim
        def dummy_func(x):
            return 0.59 * torch.log(-x + 1.22) + 0.88

        def not_dummy_func(x):
            return -0.59 * torch.log(-x + 1.22) + 0.12

        not_dummy_scale_factor = not_dummy_func(text_scores)
        not_dummy_scale_factor = not_dummy_scale_factor.expand(-1, text_feat.size(1)) #(bs, max_seq_len)
        not_dummy_scale_factor = not_dummy_scale_factor.unsqueeze(1).expand(-1, img_feat.size(1), -1)

        dummy_scale_factor = dummy_func(text_scores)
        dummy_scale_factor = dummy_scale_factor.expand(-1, text_feat.size(1)) #(bs, max_seq_len)
        dummy_scale_factor = dummy_scale_factor.unsqueeze(1).expand(-1, self.num_queries, -1)


        # not_dummy_scale_factor = text_scores.unsqueeze(1).expand(-1, img_feat.size(1), -1)
        # dummy_scale_factor = 1 - text_scores.unsqueeze(1).expand(-1, 1, -1) 
        
        scale_factor = torch.cat([not_dummy_scale_factor, dummy_scale_factor], dim=1) #(bs, N_p + num_queries, max_seq_len)
        scaled_similarity = scale_factor * similarity_norm
        #패딩부분 무시
        scaled_similarity = scaled_similarity.masked_fill(nan_mask, float('-inf'))
        #print(scaled_similarity[0])
        
        #max 추출 
        max_per_patch, max_text_idx = scaled_similarity.max(dim=-1) # (batch_size, N_p + num_queries)
        # max_per_batch, max_patch_idx = max_per_patch.max(dim=-1, keepdim=True) #(bs, 1)
        topk_per_batch, topk_patch_idx = torch.topk(max_per_patch, self.num_queries, -1) #(bs, num_queries)

        # max_patch_onehot = torch.zeros_like( #(bs, N_p+1)
        #     max_per_patch, memory_format=torch.legacy_contiguous_format
        # ).scatter_(-1, topk_patch_idx, 1.0)
        bs, num_queries = topk_patch_idx.shape
        
        max_patch_onehot = torch.zeros(bs, num_queries, max_per_patch.size(-1), device=max_per_patch.device) #(bs, num_queries, N_p + num_queries)
        idx = topk_patch_idx.unsqueeze(-1)     # (bs, num_queries, 1)
        max_patch_onehot.scatter_(2, idx, 1.0) #(bs, num_queries, N_p + num_queries)
        
        selected_patch_feat = torch.bmm(max_patch_onehot, img_feat_with_dummy) #(bs, num_queries, embed_dim)
        selected_patch_query = self.query_proj(selected_patch_feat) #그래디언트 Nan 아님
        
        return selected_patch_query, topk_patch_idx, similarity_norm, scaled_similarity, scale_factor
        
    def forward_train(
        self,
        x_mm,
        img_feat, #encoder output의 img_feat
        img_metas,
        text_feat=None,
        text_scores=None,
        text_mask=None,
        gt_bbox=None,
        gt_mask_vertices=None,
        epoch=None
    ):
        #x_mm : image feature (bs, embed_dim', H방향 패치 수, W방향 패치 수)
        #text_feat : (bs, max_sequence_length, embed_dim') #baseline은 text_feat 아예 안 씀!!!!!
        #text_mask : text padding mask

        # feature proj to embed channels
        x_mm = self.input_proj(x_mm) #nn.Conv2d(in_channels, embed_dim, kernel_size=1) -> embed_dim을 변경
        img_masks, pos_embed = self.x_mask_pos_enc(x_mm, img_metas)  # TODO: fix the img mask #padding 전처리 들어갈 경우 필요
        
        if self.language_guided_query_selection_flag:
            query_embed, topk_patch_idx, similarity, scaled_similarity, scale_factor = self.language_guided_query_selection(text_feat, img_feat, text_scores, text_mask, epoch=epoch)
            
            #a, b = self.language_guided_query_selection_pre(text_feat, img_feat, text_scores, text_mask)
            #더미토큰 처리
            
            B, C, H, W = pos_embed.shape
            n_patch = img_feat.size(1)
            dummy_idx = topk_patch_idx >= n_patch  #(bs, num_queries)
            all_dummy_idx = dummy_idx.all(dim=1) #(bs) #num_queries개수만큼 더미가 모두 뽑혔을 때 dummy_idx는 True

            #더미 뽑힌 횟수
            num_all_dummy = all_dummy_idx.sum()
            #No-target일때 더미 뽑힌 횟수
            no_target = torch.tensor([0]*B, device=all_dummy_idx.device)
            for i, meta in enumerate(img_metas):
                for target in meta['target']:
                    if target['category_id']==-1:
                        no_target[i]=1
            #num_no_target = no_target.sum()
            num_accurate_dummy = (all_dummy_idx & (no_target == 1)).sum()
            #더미 비율
            no_target_mask = no_target.bool()
            if no_target_mask.any():
                dummy_ratio_of_no_target = dummy_idx[no_target_mask, :].sum(dim=1) /self.num_queries #(num_no_target)
                sum_dummy_ratio_of_no_target = dummy_ratio_of_no_target.sum()
            else:
                sum_dummy_ratio_of_no_target = None
            
            dummy_ratio_of_others = dummy_idx[~no_target_mask, :].sum(dim=1) /self.num_queries #(num_others)
            sum_dummy_ratio_of_others = dummy_ratio_of_others.sum()

            #scale factor 0.7기준 양상
            is_over_cross = text_scores[:, 0]>=0.7 #(bs)
            no_target_is_over_cross = is_over_cross[no_target_mask] #(num_no_target)
            all_dummy_idx_no_target = all_dummy_idx[no_target_mask] #(num_no_target)
            yes_dum_no_target_is_over_cross = no_target_is_over_cross & all_dummy_idx_no_target #(num_no_target)
            
            if no_target_is_over_cross.any():
                ratio_over_cross = yes_dum_no_target_is_over_cross.sum() /no_target_is_over_cross.sum()
            else:
                ratio_over_cross = None
            
            yes_dum_no_target_is_under_cross = (~no_target_is_over_cross) & all_dummy_idx_no_target #(num_no_target)
            if (~no_target_is_over_cross).any():
                ratio_under_cross = yes_dum_no_target_is_under_cross.sum() / (~no_target_is_over_cross).sum()
            else:
                ratio_under_cross = None

            
            dummy_dict = {'num_all_dummy': num_all_dummy, 'num_accurate_dummy': num_accurate_dummy, 'dummy_idx': dummy_idx, 'sum_dummy_ratio_of_no_target' : sum_dummy_ratio_of_no_target, 'sum_dummy_ratio_of_others' : sum_dummy_ratio_of_others, 'ratio_over_cross_blah': ratio_over_cross, 'ratio_under_cross_blah' : ratio_under_cross}
            
            # h_idx = top1_idx // W
            # w_idx = top1_idx % W

            # valid_batch = torch.arange(B, device=top1_idx.device)[~dummy_idx]
            # valid_h_idx = h_idx[~dummy_idx]
            # valid_w_idx = w_idx[~dummy_idx]
            
            # # 새 위치 임베딩 가져오기 (val_bs, C)
            # selected_pos = pos_embed[valid_batch, :, valid_h_idx, valid_w_idx] #[31, 256]
            
            # # 대체: dummy_idx=False인 경우에만 교체
            # query_embed[valid_batch, :] = selected_pos

            #diversity loss
            # Normalize to unit vectors
            normalized = F.normalize(self.dummy_token, dim=-1)  # [1, num_queries, in_channels]
            
            # Cosine similarity matrix
            #cos_sim_matrix = torch.matmul(normalized, normalized.T)  # [num_queries, num_queries]
            cos_sim_matrix = torch.matmul(normalized, normalized.transpose(1, 2))
            cos_sim_matrix = cos_sim_matrix.squeeze(0)

            
            # Target similarity: identity = 1, else = 0
            target = torch.eye(self.num_queries, device=cos_sim_matrix.device)
        
            
            # MSE loss between actual similarity and target (-1 off-diagonal, 1 diagonal)
            loss_dummy_div = F.mse_loss(cos_sim_matrix, target)

            #dummy enhance loss
            nan_mask = text_mask.bool().unsqueeze(1) #(bs, 1, max_seq_len)
            expanded_mask = nan_mask.expand(-1, similarity.size(1), -1) 
            masked_similarity = similarity.masked_fill(expanded_mask, 1)
            if no_target_mask.any():
                similarity_NT_dummy = masked_similarity[no_target_mask, -self.num_queries:, :]
                nt_dummy_loss = F.mse_loss(similarity_NT_dummy, torch.ones_like(similarity_NT_dummy))
            else:
                nt_dummy_loss = torch.tensor(0.0, device=similarity.device)
                
            if (~no_target_mask).any():  # 모두 True가 아닌 경우만 처리
                similarity_others_dummy = masked_similarity[~no_target_mask, :-self.num_queries, :] #구현 확인용(나중에 변경하기)
                others_dummy_loss = F.mse_loss(similarity_others_dummy, torch.zeros_like(similarity_others_dummy))
            else:
                others_dummy_loss = torch.tensor(0.0, device=similarity.device)
        
            #all_dummy_idx=True인 샘플들 제외하고 forwarding
            valid_idx = ~all_dummy_idx
            # valid_x_mm = x_mm[valid_idx]
            # valid_img_masks = img_masks[valid_idx]
            # valid_query_embed = query_embed[valid_idx]
            # valid_pos_embed = pos_embed[valid_idx]
            


        else:
            query_embed = self.query_embed.weight

        #num_layers 개의 디코더 레이어 통과 (한 레이어 : "self_attn" -> "norm" -> "cross_attn"  -> "norm" -> "ffn" -> "norm")
        #self_attn : Q, K, V = content_query(zeros), Q_pos, K_pos = nn.Embedding
        #cross_attn : Q = content_query(zeros), K, V = x_mm, Q_pos = nn.Embedding, K_pos = pos_embed
        device = x_mm.device
        hidden_states, _ = self.transformer(x_mm, img_masks, query_embed, pos_embed) #DetrTransformer #hidden_states: [num_decoder_layer, num_valid_sam, num_query, embed_dim]
        #hidden_states, _ = self.transformer(valid_x_mm, valid_img_masks, valid_query_embed, valid_pos_embed)
        #HEAD쿼리 & 배치 전체에 대해 동일한 head를 적용
        outputs_class = self.class_embed(hidden_states) #nn.Linear(embed_dim, num_classes + 1) #outputs_class: [num_decoder_layer, num_valid_sam, num_query, 2]
        outputs_coord = self.bbox_embed(hidden_states) #MLP(input_dim=embed_dim, hidden_dim=embed_dim, output_dim=4, num_layers=3) #outputs_coord: [num_decoder_layer, num_valid_sam, num_query, 4]

        #더미토큰 처리
        if self.language_guided_query_selection_flag:
            dummy_mask = dummy_idx.unsqueeze(0).unsqueeze(-1) ##(1, bs, num_queries, 1)
            replacement_class = torch.tensor([-1000.0, 0.0], device=device)
            replacement_box = torch.tensor([-1000.0, -1000.0, -1000.0, -1000.0], device=device)

            #num_decoder_layer, batch_size, num_query = 3, all_dummy_idx.shape[0], query_embed.shape[1] #hard coding

            replacement_class = replacement_class.view(1, 1, 1, 2).expand_as(outputs_class)
            replacement_box = replacement_box.view(1, 1, 1, 4).expand_as(outputs_coord)
            
            # outputs_class = replacement_class.view(1, 1, 1, 2).expand(num_decoder_layer, batch_size, num_query, 2).clone()
            # outputs_coord = replacement_box.view(1, 1, 1, 4).expand(num_decoder_layer, batch_size, num_query, 4).clone()

            # if all_dummy_idx.sum()!=x_mm.shape[0]:
            #     # valid한 위치에 valid한 결과 삽입
            #     outputs_class[:, valid_idx, :, :] = valid_outputs_class
            #     outputs_coord[:, valid_idx, :, :] = valid_outputs_coord
            outputs_class = torch.where(dummy_mask, replacement_class, outputs_class) #[num_decoder_layer, bs, num_query, 2]
            outputs_coord = torch.where(dummy_mask, replacement_box, outputs_coord) #[num_decoder_layer, bs, num_query, 4]

        outputs_coord = outputs_coord.sigmoid()
    
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]} #pred_logits: [bs, num_query, 2], pred_boxes: [bs, num_query, 4]
        targets = self.prepare_targets(gt_bbox, img_metas) #gt logit, gt 좌표 준비 {"labels": [n], "boxes": [1, n, 4]}의 리스트
        
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
            
        loss_dict = self.criterion(output, targets) #{"loss_class": loss_class, "loss_giou" : loss_giou, "loss_bbox": loss_bbox}

        #class loss 실험
        if all_dummy_idx.sum()!=x_mm.shape[0]:
            valid_outputs_coord = outputs_coord[:, valid_idx, :, :]
            valid_outputs_class = outputs_class[:, valid_idx, :, :]
            #valid_outputs_coord = valid_outputs_coord.sigmoid()
            valid_output = {"pred_logits": valid_outputs_class[-1], "pred_boxes": valid_outputs_coord[-1]}
            if self.aux_loss:
                valid_output["aux_outputs"] = self._set_aux_loss(valid_outputs_class, valid_outputs_coord)
            targets_valid = [x for x, m in zip(targets, valid_idx) if m]
            loss_dict_valid = self.criterion(valid_output, targets_valid)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict_valid.keys():
                if k in weight_dict:
                    loss_dict_valid[k] *= weight_dict[k]
            loss_dict_valid["loss_det"] = sum(loss_dict_valid.values()) #최종 loss (giou + l1 + ce)
        else:
            loss_dict_valid = {
                'loss_class': torch.tensor(0.0, device=device),
                'loss_bbox': torch.tensor(0.0, device=device),
                'loss_giou': torch.tensor(0.0, device=device),
                'loss_det': torch.tensor(0.0, device=device)
            }

        weight_dict = self.criterion.weight_dict
        for k in loss_dict.keys():
            if k in weight_dict:
                loss_dict[k] *= weight_dict[k]
        loss_dict["loss_det"] = sum(loss_dict.values()) #최종 loss (giou + l1 + ce)
        if self.language_guided_query_selection_flag:
            loss_dict['dummy_token_diversity_loss'] = loss_dummy_div
            loss_dict['nt_dummy_loss'] = nt_dummy_loss
            loss_dict['others_dummy_loss'] = others_dummy_loss

        else:
            dummy_dict, similarity, scaled_similarity, scale_factor = None, None, None, None
        return loss_dict, loss_dict_valid, output, dummy_dict, similarity, scaled_similarity, scale_factor

        # proj_queries = F.normalize(self.contrastive_align_projection_image(logits), p=2, dim=-1)
        # proj_tokens = F.normalize(
        #     self.contrastive_align_projection_text(memory_cache["text_memory"]).transpose(0, 1),
        #     p=2,
        #     dim=-1,
        # )
        # out.update(
        #     {
        #         "proj_queries": proj_queries[-1],
        #         "proj_tokens": proj_tokens,
        #         "tokenized": memory_cache["tokenized"],
        #     }
        # )
        # assert proj_tokens is not None and proj_queries is not None
        # out["aux_outputs"] = [
        #     {
        #         "pred_logits": a,
        #         "pred_boxes": b,
        #         "proj_queries": c,
        #         "proj_tokens": proj_tokens,
        #         "tokenized": memory_cache["tokenized"],
        #     }
        #     for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], proj_queries[:-1])
        # ]

        # loss_dict = {}
        # if self.criterion is not None:
        #     loss_dict.update(self.criterion(out, targets, positive_map))

        # loss_ce = self.loss(logits, targets, with_bbox=with_bbox, with_mask=with_mask)

    def forward_test(self, x_mm, img_feat, img_metas, text_scores=None, text_mask = None, text_feat=None, with_bbox=False, with_mask=False):
        x_mm = self.input_proj(x_mm)
        img_masks, pos_embed = self.x_mask_pos_enc(x_mm, img_metas)  # TODO: fix the img mask
        
        if self.language_guided_query_selection_flag:
            query_embed, topk_patch_idx, similarity, scaled_similarity = self.language_guided_query_selection(text_feat, img_feat, text_scores, text_mask, epoch=epoch)
            B, C, H, W = pos_embed.shape
            n_patch = img_feat.size(1)
            dummy_idx = topk_patch_idx == n_patch  #(bs, num_queries)
            dummy_idx = dummy_idx.all(dim=1) #(bs)
        else:
            query_embed = self.query_embed.weight
            
        hidden_states, _ = self.transformer(x_mm, img_masks, query_embed, pos_embed)

        outputs_class = self.class_embed(hidden_states)
        outputs_coord = self.bbox_embed(hidden_states).sigmoid()

        #더미토큰 처리
        replacement_class = torch.tensor([-1000.0, 0.0], device=outputs_class.device)
        outputs_class[:, dummy_idx, :, :] = replacement_class.view(1, 1, 1, 2).expand(outputs_class.size(0), dummy_idx.sum(), self.num_queries, 2)
        replacement_box = torch.tensor([-1000.0, -1000.0, -1000.0, -1000.0], device=outputs_coord.device)
        outputs_coord[:, dummy_idx, :, :] = replacement_box.view(1, 1, 1, 4).expand(outputs_coord.size(0), dummy_idx.sum(), self.num_queries, 4)
        outputs_coord = outputs_coord.sigmoid()

        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}

        return output

    def inference(self, box_cls, box_pred, image_sizes):
        """Inference function for DETR

        Args:
            box_cls (torch.Tensor): tensor of shape ``(batch_size, num_queries, K)``.
                The tensor predicts the classification probability for each query.
            box_pred (torch.Tensor): tensors of shape ``(batch_size, num_queries, 4)``.
                The tensor predicts 4-vector ``(x, y, w, h)`` box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # For each box we assign the best class or the second best if the best on is `no_object`.(클래스 차원의 마지막 요소를 항상 non-object로 씀)
        scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1) #1. no-object class 제외하고 가장 높은 클래스 선택 GREC) 객체가 있을 confidence score, labels(항상 0)

        #scores: (batch_size, num_queries)
        #labels: (batch_size, num_queries)

        #각 샘플마다 저장
        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
            zip(scores, labels, box_pred, image_sizes)
        ):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))  #(num_queries, 4)
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image #예측 스코어 #(num_q)
            result.pred_classes = labels_per_image #예측된 레이블 #(num_q)
            results.append(result)
        return results #리스트로 구분됨