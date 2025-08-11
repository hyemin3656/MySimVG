from torch.nn import MultiheadAttention, LayerNorm
from torchscale.component.feedforward_network import FeedForwardNetwork
import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


class ExisEncoderLayer(nn.Module):
    def __init__(self, embed_dim, self_attn=True, ffn=True):
        super().__init__()
        self.embed_dim = embed_dim #768
        self.num_heads = 8
        self.dropout_module = torch.nn.Dropout(0.7) #0.7
        self.self_attn_flag = self_attn
        if self_attn == True:
            self.self_attn = self.build_self_attention(self.embed_dim, self.num_heads)
            self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.cross_attn = self.build_cross_attention(self.embed_dim, self.num_heads)
        self.cross_attn_layer_norm = LayerNorm(self.embed_dim)

        self.ffn_flag = ffn
        if ffn:
            self.ffn_dim = embed_dim * 4
            self.activation_fn = "gelu"
            self.dropout_fn = 0.1
            self.activation_dropout_fn = 0.0
            self.layernorm_eps = 1e-5
            self.ffn = self.build_ffn(
                self.embed_dim,
                self.embed_dim,
                self.activation_fn,
                self.dropout_fn,
                self.activation_dropout_fn,
                self.layernorm_eps
            )
            self.ffn_layer_norm = LayerNorm(self.embed_dim)
            
    def build_self_attention(self, embed_dim, num_heads):
        return MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True
        )

    def build_cross_attention(self, embed_dim, num_heads):
        return MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True
        )

    def build_ffn(self, embed_dim, ffn_dim, activation_fn, dropout_fn, activation_dropout_fn, layernorm_eps):
        return FeedForwardNetwork(
            embed_dim,
            ffn_dim,
            activation_fn,
            dropout_fn,
            activation_dropout_fn,
            layernorm_eps,
            True, #sub-layer Norm: Ture -> FFN 내부에서 추가적인 Layer Norm
        )
    
    
    def residual_connection(self, x, residual):
        return residual + x # * self.alpha
    
    def forward(self, img_feature, txt_feature, text_mask=None):
            #attention
            #Q : txt_feature
            #K, V : img_feature
            #attn_mask : (batch_size*num_heads, target seq len, source seq len)
            key_padding_mask = text_mask.bool()  #(bs, max_seq_len+1)
            #key_padding_mask = key_padding_mask.masked_fill(key_padding_mask.to(torch.bool), -1e8)
            mask_q = text_mask.unsqueeze(1).unsqueeze(-1) # (batch_size, max_seq_len) -> (batch_size, 1, max_seq_len, 1)
            #--------------self-att, Add&Norm----------------#
            if self.self_attn_flag == True:
                # mask_k = text_mask.unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, max_seq_len)
                # self_attention_mask = mask_q | mask_k  # (batch_size, 1, max_seq_len, max_seq_len) # 둘 중 하나라도 padding이면 mask
                # self_attention_mask = self_attention_mask.expand(-1, self.num_heads, -1, -1).flatten(0, 1).bool() # (batch_size*num_heads, seq_len, seq_len)
                # self_attention_mask = self_attention_mask.masked_fill(self_attention_mask.to(torch.bool), -1e8)
                
                residual = txt_feature
                x = self.self_attn_layer_norm(txt_feature) #(bs, max_seq_len+1, embed_dim)
                x = x * (1 - text_mask.unsqueeze(-1).type_as(x)) #패딩된부분 무시 #(bs, max_seq_len+1, 1)
                x, attn_map = self.self_attn( 
                    query=x,
                    key=x,
                    value=x,
                    key_padding_mask=key_padding_mask,
                    attn_mask= None
                )
                x = self.dropout_module(x)
                
                x = self.residual_connection(x, residual)
                residual = x
                x = self.cross_attn_layer_norm(x)
            else:
                residual = txt_feature
                x = self.cross_attn_layer_norm(txt_feature)
                
            #--------------cross-att, Add&Norm---------------#
            # cross_attention_mask = mask_q.repeat(1, self.num_heads, 1, img_feature.shape[-2])
            # cross_attention_mask = cross_attention_mask.flatten(0, 1).bool()
            # cross_attention_mask = cross_attention_mask.masked_fill(cross_attention_mask.to(torch.bool), -1e8)
    
            # if self.normalize_before:
            #     x = self.self_attn_layer_norm(x)
            x = x * (1 - text_mask.unsqueeze(-1).type_as(x)) #패딩된부분 무시
            x, attn_map = self.cross_attn( 
                query=x,
                key=img_feature,
                value=img_feature,
                attn_mask=None 
            )
            x = self.dropout_module(x)

            # if self.drop_path is not None:
            #     x = self.drop_path(x) #Residual Connection을 랜덤하게 비활성화

            x = self.residual_connection(x, residual)
            # if not self.normalize_before: 
            #     x = self.self_attn_layer_norm(x) #self-att 이후에 LayerNorm

            if self.ffn_flag == True:
                residual = x
                x = self.ffn_layer_norm(x)
                #--------------FFN---------------#
                # if self.normalize_before:
                #     x = self.final_layer_norm(x)
                x = self.ffn(x) 
    
                # if self.drop_path is not None:
                #     x = self.drop_path(x)

                x = self.residual_connection(x, residual)
                # if not self.normalize_before:
                #     x = self.final_layer_norm(x)
            return x
    
class ExisEcoder(nn.Module):
    def __init__(self, embed_dim, sentence_token_flag):
        super().__init__()
        self.embed_dim = embed_dim
        self.sentence_token_flag = sentence_token_flag
        self.num_encoder_layers = 3 #추후 변경
            
        #레이어 정의
        self.layers = nn.ModuleList([])
        for i in range(self.num_encoder_layers):
            self.layers.append(
                self.build_encoder_layer(embed_dim)
            )
        if sentence_token_flag==True:
            #sentence level token
            self.sentence_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.lp = nn.Linear(embed_dim*2, 1)

    def build_encoder_layer(self, embed_dim):
        layer = ExisEncoderLayer(embed_dim)
        return layer
    
    def forward(self, img_feature, txt_feature, text_mask=None, img_metas=None):
        # def debug_tensor(name, tensor, verbose=True):
        #     has_nan = torch.isnan(tensor).any().item()
        #     has_inf = torch.isinf(tensor).any().item()
        
        #     if has_nan or has_inf:
        #         print(f"\n[⚠️ WARNING] {name} has", end=' ')
        #         if has_nan: print("NaN", end=' ')
        #         if has_inf: print("Inf", end=' ')
        #         print("values!")
        
        #         print(f"  → shape: {tensor.shape}")
        #         if verbose:
        #             print(f"  → min: {tensor.min().item()}, max: {tensor.max().item()}")
        #         return True
        #     return False
        
        x = txt_feature
        if self.sentence_token_flag==True:
            sentence_token = self.sentence_token.expand(
                x.size(0), -1, -1
            ) 
            x = torch.cat((x, sentence_token), dim=1) #(bs, max_seq_len+1, embed_dim)
            #mask
            sen_token_padding_mask = torch.zeros((x.size(0), 1), dtype=x.dtype, device=x.device)
            text_mask = torch.cat([text_mask, sen_token_padding_mask], dim=1) #(bs, max_seq_len+1)
        
        for idx, layer in enumerate(self.layers):
            x = layer(
                img_feature,
                x,
                text_mask
            )

        if self.sentence_token_flag==True:
            sent_feat = x[:, -1, :]
            return sent_feat #(bs, embed_dim)
        else:
        #x:(bs, max_seq_len, embed_dim)
        #text_mask:(bs, max_seq_len)

            muti_feature = torch.cat((txt_feature, x), dim =-1) #(bs, max_seq_len, embed_dim*2)
            #LP
            exis_scores = self.lp(muti_feature) #(bs, max_seq_len, 1)
            # print('exis', exis_scores)

            exis_probs = torch.sigmoid(exis_scores)
            # print(exis_probs)
            exis_probs = exis_probs.squeeze(-1) # bs, max_seq_len
            # #compute loss-----
            # #GT binary target 생성
            # gt_scores = torch.tensor([0 if img_meta["target"][0]["category_id"] == -1 else 1 for img_meta in img_metas], device=exis_probs.device).float() #(bs)
            # #masked predict
            # exis_probs_masked = exis_probs.masked_fill(text_mask.bool(), float('inf'))
            # min_probs, min_idx = torch.min(exis_probs_masked, dim=1)  # (bs,)
            
            # # BCE loss 계산
            # loss = F.binary_cross_entropy(min_probs, gt_scores)
    
            #시각화
            # exis_probs_masked_vis = exis_probs.masked_fill(text_mask.bool(), float(0.0))  #bs, max_seq_len
            
            # gt_scores_bool = gt_scores.bool()
            # others = exis_probs_masked_vis[gt_scores_bool, :]
            # no_target = exis_probs_masked_vis[~gt_scores_bool, :]
                    
            # # 저장용 폴더 생성 (선택)
            # os.makedirs("heatmaps", exist_ok=True)
            
            # # 저장 함수
            # def plot_heatmap(data, title, filename):
            #     plt.figure(figsize=(10, 6))
            #     plt.imshow(data, aspect='auto', cmap='viridis')
            #     plt.colorbar(label='Score')
            #     plt.title(title)
            #     plt.xlabel('Score Index')
            #     plt.ylabel('Sample Index')
            #     plt.tight_layout()
            #     plt.savefig(filename)   # 이미지로 저장
            #     plt.close()             # 메모리 정리 (안 하면 누적됨)
            
            # # 히트맵 저장
            # plot_heatmap(others.cpu().numpy(), title='Score Heatmap - Others Samples', filename='heatmaps/others_heatmap.png')
            # plot_heatmap(no_target.cpu().numpy(), title='Score Heatmap - No-target Samples', filename='heatmaps/no_target_heatmap.png')
            return exis_probs
