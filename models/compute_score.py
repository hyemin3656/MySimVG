from torch.nn import MultiheadAttention, LayerNorm
from torchscale.component.feedforward_network import FeedForwardNetwork
import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from detrex.layers.position_embedding import PositionEmbeddingLearned, PositionEmbeddingSine

class ExisEncoderLayer(nn.Module):
    def __init__(self, embed_dim, self_attn=True, ffn=True):
        super().__init__()
        self.embed_dim = embed_dim #768
        self.num_heads = 8
        self.dropout_module = torch.nn.Dropout(0.7) #0.7
        self.self_attn_flag = self_attn
        if self_attn == True:
            self.self_attn = self.build_self_attention(self.embed_dim, self.num_heads)
            self.self_attn_layer_norm = LayerNorm(self.embed_dim) #feature축에서만 정규화
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

    def forward(self, img_feature, txt_feature, text_mask, img_pos_embed=None, img_masks=None):
        key_padding_mask = text_mask.bool()  #(bs, max_seq_len+1)
        if img_masks:
            key_padding_mask_img = img_masks.bool()  #(bs, h*w)
        else:
            key_padding_mask_img=None

        x = txt_feature  # (bs, max_seq_len+1, embed_dim) #(pos_embed, 적용됨)

        #-------------- Self-Attention ----------------#
        if self.self_attn_flag:
            residual = x

            self_attn_out, self_attn_map = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=key_padding_mask,
                attn_mask=None
            )

            self_attn_out = self.dropout_module(self_attn_out)
            x = self.residual_connection(self_attn_out, residual)  # Add
            x = self.self_attn_layer_norm(x)  # Norm

        #-------------- Cross-Attention ----------------#
        residual = x

        #query = text, key = image
        if img_pos_embed:
            k = img_feature + img_pos_embed
        else:
            k = img_feature
        v = img_feature  # value는 pos 안 더함

        cross_attn_out, cross_attn_map = self.cross_attn(
            query=x,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask_img,
            attn_mask=None
        )

        cross_attn_out = self.dropout_module(cross_attn_out)
        x = self.residual_connection(cross_attn_out, residual)  # Add
        x = self.cross_attn_layer_norm(x)  # Norm

        #-------------- FFN ----------------#
        if self.ffn_flag:
            residual = x
            ffn_out = self.ffn(x)
            ffn_out = self.dropout_module(ffn_out)
            x = self.residual_connection(ffn_out, residual)  # Add
            x = self.ffn_layer_norm(x)  # Norm
        return x
    
class ExisEcoder(nn.Module):
    def __init__(self, in_channels, sentence_token_flag):
        super().__init__()
        self.in_channels = in_channels
        self.sentence_token_flag = sentence_token_flag
        #--------------hyperparam-------------#
        self.num_encoder_layers = 3 #추후 변경
        self.embed_dim = 768
        self.projection = False
        self.pos_enc = False
        if self.projection:
            if self.pos_enc:
                self.img_proj = nn.Conv2d(in_channels, self.embed_dim, kernel_size=1) #image feature projection
            else:
                self.img_proj = nn.Linear(in_channels, self.embed_dim)
            self.txt_proj = nn.Linear(in_channels, self.embed_dim) #text feature projection
        else:
            self.embed_dim = in_channels
        if self.pos_enc:
            self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=self.embed_dim // 2,
            temperature=10000,
            normalize=True,
        )

        #레이어 정의
        self.layers = nn.ModuleList([])
        for i in range(self.num_encoder_layers):
            self.layers.append(
                self.build_encoder_layer(self.embed_dim)
            )
        if sentence_token_flag==True:
            #sentence level token
            self.sentence_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            #sentence token position embedding
            if self.pos_enc:
                self.sent_token_pos_embedding = nn.Parameter(torch.randn(1, self.embed_dim))

        else:
            self.lp = nn.Linear(self.embed_dim*2, 1)

    def build_encoder_layer(self, embed_dim):
        layer = ExisEncoderLayer(embed_dim)
        return layer
    
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
    
    def sinusoidal_positional_embedding(self, token_sequence_size, token_embedding_dim, n=10000.0):

        if token_embedding_dim % 2 != 0:
            raise ValueError("Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={:d})".format(token_embedding_dim))

        T = token_sequence_size
        d = token_embedding_dim #d_model=head_num*d_k, not d_q, d_k, d_v

        positions = torch.arange(0, T).unsqueeze_(1)
        embeddings = torch.zeros(T, d)

        denominators = torch.pow(n, 2*torch.arange(0, d//2)/d) # 10000^(2i/d_model), i is the index of embedding
        embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
        embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))

        return embeddings
    
    def forward(self, x_mm, txt_feature, text_mask=None, img_metas=None):
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
        
        #input projection
        if self.projection:
            x_mm = self.img_proj(x_mm)
            txt_feature = self.txt_proj(txt_feature)
        if self.pos_enc:
            assert len(xmm.shape)!=4
            #position embedding
            img_masks, img_pos_embed = self.x_mask_pos_enc(x_mm, img_metas)
            txt_pos_embed = self.sinusoidal_positional_embedding(txt_feature.shape[1], self.embed_dim).to(txt_feature.device)  # [max_seq_len, embed_dim]
            #reshape
            bs, c, h, w = x_mm.shape
            x_mm = x_mm.view(bs, c, -1).permute(0, 2, 1)  # [bs, h*w, c]
            img_pos_embed = img_pos_embed.view(bs, c, -1).permute(0, 2, 1)  # [bs, h*w, c]
            txt_pos_embed = txt_pos_embed.unsqueeze(0).repeat(bs, 1, 1) #(bs, max_seq_len, c)
            img_masks = img_masks.view(bs, -1)  # [bs, h, w] -> [bs, h*w]
        else:
            img_masks=None
            img_pos_embed=None

        x = txt_feature
        if self.sentence_token_flag==True:
            sentence_token = self.sentence_token.expand(
                x.size(0), -1, -1
            ) 
            x = torch.cat((x, sentence_token), dim=1) #(bs, max_seq_len+1, embed_dim)
            #mask
            sen_token_padding_mask = torch.zeros((x.size(0), 1), dtype=x.dtype, device=x.device)
            text_mask = torch.cat([text_mask, sen_token_padding_mask], dim=1) #(bs, max_seq_len+1)
            if self.pos_enc:
                #pos embedding
                sent_token_pos_embedding = self.sent_token_pos_embedding.unsqueeze(0).repeat(bs, 1, 1)
                txt_pos_embed = torch.cat([txt_pos_embed, sent_token_pos_embedding], dim=1) #(bs, max_seq_len+1, c)

                #text pos embedding 적용
                x = x + txt_pos_embed

        
        for idx, layer in enumerate(self.layers):
            x = layer(
                x_mm,
                x,
                text_mask,
                img_pos_embed,
                img_masks
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
