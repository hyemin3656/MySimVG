from torchscale.component.multihead_attention import MultiheadAttention
from torchscale.component.feedforward_network import FeedForwardNetwork
import torch.nn
import math
from fairscale.nn import checkpoint_wrapper, wrap

class ExisEcoderLayer(nn.Moduel):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.exis_embed_dim
        self.cross_attn = self.build_cross_attention(self.embed_dim, args)
        self.dropout_module = torch.nn.Dropout(args.dropout)
        self.fnn = self.build_ffn(self.embed_dim, self.args)


    def build_ffn(self, embed_dim, args):
        return FeedForwardNetwork(
            embed_dim,
            self.ffn_dim,
            args.activation_fn,
            args.dropout,
            args.activation_dropout,
            args.layernorm_eps,
            args.subln, #sub-layer Norm: Ture -> FFN 내부에서 추가적인 Layer Norm
        )
    
    def build_cross_attention(self, embed_dim, args):
        return MultiheadAttention(
            args,
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=False,
            encoder_decoder_attention=False,
            subln=args.subln,
        )
    
    def residual_connection(self, x, residual):
        return residual + x # * self.alpha
    
    def forward(self, txt_feature, img_feature, attn_mask=None, rel_pos=None, incremental_state=None):
            #attention
            #Q : txt_feature
            #K, V : img_feature
            #--------------cross-att, Add&Norm---------------#
            residual = img_feature
            # if self.normalize_before:
            #     x = self.self_attn_layer_norm(x) #self-att 이전에 LayerNorm
            x, attn_map = self.cross_attn( 
                query=txt_feature,
                key=img_feature,
                value=img_feature,
                attn_mask=attn_mask,
                rel_pos=rel_pos,
                incremental_state=incremental_state,
            )
            x = self.dropout_module(x)

            # if self.drop_path is not None:
            #     x = self.drop_path(x) #Residual Connection을 랜덤하게 비활성화

            x = self.residual_connection(x, residual)
            # if not self.normalize_before: 
            #     x = self.self_attn_layer_norm(x) #self-att 이후에 LayerNorm

            residual = x
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
    
class ExisEcoderr(nn.Moduel):
    def __init__(self, args):
        self.args = args
        self.dropout_module = torch.nn.Dropout(args.dropout)
        self.embed_dim = args.exis_embed_dim
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(self.embed_dim) #학습 안정성 위함
            
        #레이어 정의
        self.layers = nn.ModuleList([])
        for i in range(args.encoder_layers):
            self.layers.append(
                self.build_encoder_layer(args)
            )
        self.num_layers = len(self.layers)

    def build_encoder_layer(
        self, args, depth, is_moe_layer=False, is_encoder_decoder=False
    ):
        layer = ExisEcoderLayer(
            args,
        )
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        if args.fsdp:
            layer = wrap(layer) #메모리 최적화
        return layer
    
    def forward(self, txt_feature, img_feature):
        x = img_feature
        for idx, layer in enumerate(self.layers):
            x = layer(
                txt_feature,
                x,
            )
        
        #concat
        muti_feature = torch.cat((txt_feature, x), dim =-1)
        
        #MLP
        lp = nn.Linear(muti_feature.shape[-1], 1)
        exis_score = lp(muti_feature)
        
        return exis_score
    
