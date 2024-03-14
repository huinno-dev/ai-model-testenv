import torch.nn as nn
import torch.nn.functional as F
import torch
import math


def get_model(**model):
    # Default or Not
    num_class = model['num_class'] if 'num_class' in model else 4
    dim_model = model['dim_model'] if 'dim_model' in model else 128
    dim_ff = model['dim_ff'] if 'dim_ff' in model else 128
    n_encoder = model['n_encoder'] if 'n_encoder' in model else 4
    n_decoder = model['n_decoder'] if 'n_decoder' in model else 6
    n_head = model['n_head'] if 'n_head' in model else 8
    len_resample = model['len_resample'] if 'len_resample' in model else 64

    model = Transformer(num_class=num_class, dim_model=dim_model, dim_ff=dim_ff,
                        n_encoder=n_encoder, n_decoder=n_decoder, n_head=n_head, dropout=0.1, len_resample=len_resample)

    # model.apply(initializer)
    return model


class Transformer(nn.Module):
    def __init__(self, num_class: int = 4, dim_model: int = 512, dim_ff: int = 2048, len_resample=128,
                 n_encoder: int = 2, n_decoder: int = 4, n_head: int = 8, dropout: float = 0.1):
        super(Transformer, self).__init__()
        self.n_head = n_head
        self.n_encoder = n_encoder
        self.n_decoder = n_decoder
        self.dim_model = dim_model
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.len_resample = len_resample

        self.src_emb = Embedding(dim_embed=dim_model, expand_ch_dim=64)
        # self.trg_emb = Embedding(in_ch=4, n_unit=512, dim_embed=64, dropout=0.1, max_length=1, emb_type='transformer')
        self.trg_emb = nn.Embedding(num_class + 2, dim_model)   # class embedding with <pad>, <unk>

        encoders = [AttentionBlock('encoder', n_head, dim_model, dim_hidden=dim_ff) for _ in range(self.n_encoder)]
        self.encoders = nn.ModuleList(encoders)
        decoders = [AttentionBlock('decoder', n_head, dim_model, dim_hidden=dim_ff) for _ in range(self.n_decoder)]
        self.decoders = nn.ModuleList(decoders)

        self.linear = nn.Linear(dim_model, num_class + 1)

    # 소스 문장의 <pad> 토큰에 대하여 마스크(mask) 값을 0으로 설정
    def pad_masking(self, unit):
        # unit: [batch_size, len_seq, len_resample]
        # mask: [batch_size, 1, 1, len_seq]
        mask = torch.where(unit.mean(-1) != 0 * self.len_resample, 1, 0).unsqueeze(1).unsqueeze(2)
        return mask

    @staticmethod
    def diag_masking(mask):
        # mask: [batch_size, 1, 1, len_seq]
        # canvas: [batch_size, 1, len_seq, len_seq]

        # canvas = torch.eye(mask.shape[-1]).unsqueeze(0).unsqueeze(1)
        canvas = 1 - torch.eye(mask.shape[-1]).unsqueeze(0).unsqueeze(1)
        # canvas = 1 - torch.triu(torch.ones((mask.shape[-1], mask.shape[-1]))).unsqueeze(0).unsqueeze(1)
        # canvas = torch.zeros_like(canvas)

        canvas = canvas.to(mask.device) * mask
        return canvas

    def forward(self, x: tuple):
        # x: (src, trg); src -> [batch_size, len_seq, len_resample], trg -> [batch_size, seq_len]

        if len(x) != 2: raise IndexError
        unit, cls = x
        pad_mask = self.pad_masking(unit)  # gen_mask for src and trg.
        trg_mask = self.diag_masking(pad_mask)

        unit = self.src_emb(unit)
        for i_enc in range(self.n_encoder):
            unit = self.encoders[i_enc](unit, mask=pad_mask)

        cls = self.trg_emb(cls)
        for i_dec in range(self.n_decoder):
            cls = self.decoders[i_dec](cls, mask=trg_mask, enc_attn=unit, enc_mask=pad_mask)

        if self.n_decoder == 0:
            out = self.linear(unit)
        else:
            out = self.linear(cls)        # use soft-max function to greedy decoding.
        return out


class AttentionBlock(nn.Module):
    def __init__(self, block_type: str, n_head: int, dim_model: int, dim_hidden: int, drop: float = 0.3):
        super(AttentionBlock, self).__init__()
        self.block_type = block_type
        self.n_head = n_head
        self.dim_model = dim_model
        self.dim_hidden = dim_hidden
        self.drop = drop

        self.attn = MultiHeadedAttention(n_head, dim_model)
        self.layer_norm = nn.LayerNorm(dim_model)
        self.layer_norm_out = nn.LayerNorm(dim_model)
        if block_type == 'decoder':
            self.attn_ = MultiHeadedAttention(n_head, dim_model)
            self.layer_norm_ = nn.LayerNorm(dim_model)
        self.ff = nn.Sequential(*[nn.Linear(dim_model, dim_hidden), nn.GELU(), nn.Linear(dim_hidden, dim_model)])
        self.dropout = nn.Dropout(self.drop)

    def forward(self, x, mask=None, enc_attn=None, enc_mask=None):

        self_attn = self.attn(x, mask)
        norm = self.layer_norm(x + self.dropout(self_attn))
        # norm = self.layer_norm(self_attn)

        if self.block_type == 'decoder':
            self_attn = self.attn_((norm, enc_attn, enc_attn), enc_mask)
            # norm = self.layer_norm_(self_attn)
            norm = self.layer_norm_(x + self.dropout(self_attn))

        feed = self.ff(norm)
        out = self.layer_norm_out(norm + self.dropout(feed))
        # out = self.layer_norm_out(feed)
        return out


class Embedding(nn.Module):
    """
        Inherit nn.Module first for Method Resolution Order(MRO).
        BERT Embedding which is consisted with under features
            1. TokenEmbedding : normal embedding matrix
            2. PositionalEmbedding : adding positional information using sin, cos
            2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
            sum of these features are output of Embedding
    """

    def __init__(self, dim_embed: int, expand_ch_dim: int = 64):
        super().__init__()
        self.dim_embed = dim_embed
        self.expand_ch_dim = expand_ch_dim
        self.norm = nn.BatchNorm1d(expand_ch_dim)
        # self.token = nn.Embedding(len_seq, dim_embed)
        self.ch_pool_fore = nn.AdaptiveAvgPool1d(expand_ch_dim)
        self.token = SKConv1D(ch_in=expand_ch_dim, ch_out=expand_ch_dim, kernel_size=(3, 5, 7, 9), out_size=dim_embed)
        self.sinusoidal_position(100, dim_embed)
        self.dropout = nn.Dropout(p=0.1)

    def sinusoidal_position(self, max_length, dim_emb):
        # Compute the positional encodings once in log space.
        position_encode = torch.zeros(max_length, dim_emb).float()
        position_encode.require_grad = False

        position = torch.arange(0, max_length).float().unsqueeze(1)
        div_term = (torch.arange(0, dim_emb, 2).float() * -(math.log(10000.0) / dim_emb)).exp()

        position_encode[:, 0::2] = torch.sin(position * div_term)
        position_encode[:, 1::2] = torch.cos(position * div_term)

        pe = position_encode.unsqueeze(0)
        self.register_buffer('pe', pe)  # no update (not trainable)

        # return self.pe[:, :x.size(1)]

    def forward(self, sequence):
        seq = self.ch_pool_fore(sequence.transpose(2, 1)).transpose(2, 1)
        tok = self.token(self.norm(seq))
        tok = nn.AdaptiveAvgPool1d(sequence.shape[-2])(tok.transpose(2, 1)).transpose(2, 1)
        emb = tok + self.pe[:, :tok.size(1)]
        return self.dropout(emb)


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    Usage
        1. Encoder block ( input x w/o mask )
        2. Decoder 1st attn block ( masked; input x w/ mask )
        3. Decoder 2nd attn block ( input tuple consists of q, k, v w/o mask )
    """

    def __init__(self, n_head, dim_model, dropout=0.1):
        super().__init__()
        assert dim_model % n_head == 0

        # Assume that dim_value always equals to dim_key
        self.dim_key = dim_model // n_head
        self.n_head = n_head

        self.q_linear = nn.Linear(dim_model, dim_model)
        self.k_linear = nn.Linear(dim_model, dim_model)
        self.v_linear = nn.Linear(dim_model, dim_model)
        self.output_linear = nn.Linear(dim_model, dim_model)
        self.attention = Attention(dropout)

    def forward(self, x, mask=None):
        """
        Shape of x -> [ B(batch_size), S(seq_len), D(dim) ]
        Shape of mask -> [ B(batch_size), S(seq_len) ]
        """

        # 1) Do all the linear projections in batch.
        if isinstance(x, tuple):        # source-to-target attention
            if len(x) == 3:
                # Since x is tuple(and has 3 lengths), assume that k, v is from the encoder.
                q, k, v = x
                q, k, v = self.q_linear(q), self.k_linear(k), self.v_linear(v)
                batch_size = q.shape[0]
            else:
                raise TypeError
        else:                           # self attention
            batch_size = x.shape[0]
            q, k, v = self.q_linear(x), self.k_linear(x), self.v_linear(x)
        # 2) Split dim_model into (n_head, dim_key). Shape of key is [ B, S, n_head, dim_key ]
        q = q.reshape(batch_size, -1, self.n_head, self.dim_key)
        k = k.reshape(batch_size, -1, self.n_head, self.dim_key)
        v = v.reshape(batch_size, -1, self.n_head, self.dim_key)
        # 3) Swap S axis and n_head axis. Shape of key is [ B, n_head, S, dim_key ]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # 4) Apply attention on all the projected vectors(q, k, v) in batch. Shape of attention is [ B, n_head, S, S' ]
        # S' is dim of sentence after softmax.
        attn, score = self.attention(q, k, v, mask=mask)

        # 5) Swap n_head axis and S axis. Shape of attn is [ B, S, n_head, S' ]
        # After transpose, address of the value in tensor might be disordered. "contiguous" makes the address smooth.
        attn = attn.transpose(1, 2).contiguous()
        # 6) Merge two axes of n_head and S. Shape of attn is [ B, S, dim_model ]
        attn = attn.reshape(batch_size, -1, self.n_head * self.dim_key)
        # 7) Final linear operation.
        out = self.output_linear(attn)
        return out


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def __init__(self, dropout: float = 0.1):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Shape of inputs -> [ B(batch_size), n_head, S(seq_len), dim_key ]
        Shape of mask -> [ B, S ]
        """
        # 1) Compute matching score of query and key. Now the shape of scores is [ B, n_head, S, S ]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        # 2) If the value of mask is 0, the score is negative inf.
        if mask is not None:
            # scores -= 1e9 * (1.0 - mask[:, None, None, :].float())
            scores = scores.masked_fill(mask == 0, -1e9)  # masking
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        attn = torch.matmul(scores, value)

        return attn, scores


class SKConv1D(nn.Module):
    def __init__(self, ch_in: int, ch_out: int = None, kernel_size: list or tuple = None,
                 out_size: int = 32, ch_ratio: int = 2):
        super(SKConv1D, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out or ch_out
        self.kernel_size = kernel_size or [3, 5]
        self.kernel_valid()
        self.convs = nn.ModuleList([nn.Conv1d(ch_in, ch_out, kernel_size=k, groups=ch_in) for k in kernel_size])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fuse_layer = self.fuse(ch_out, ch_out//ch_ratio)
        self.scoring = nn.Conv1d(ch_out//ch_ratio, ch_out * len(kernel_size), kernel_size=(1, ), groups=ch_in//ch_ratio)
        self.selection = nn.Softmax(-1)
        self.spatial_pool = nn.AdaptiveAvgPool1d(out_size)

    def kernel_valid(self):
        for k in self.kernel_size: assert k % 2

    @staticmethod
    def fuse(ch_in, ch_out):
        return nn.Sequential(*[nn.Conv1d(ch_in, ch_out, kernel_size=(1, )), nn.BatchNorm1d(ch_out), nn.ReLU()])

    def forward(self, x):
        feats = [c(x) for c in self.convs]
        mixture = self.pool(sum(feats))
        fused = self.fuse_layer(mixture)
        score = self.scoring(fused).reshape(fused.shape[0], self.ch_out, 1, len(self.kernel_size))
        score = self.selection(score)
        res = sum([feats[i]*score[:, :, :, i] for i in range(len(self.kernel_size))])
        return self.spatial_pool(res)
