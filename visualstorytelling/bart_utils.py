import torch
from torch import nn
from torch.nn import functional as F
import random


class VisualEncoder(nn.Module):
    def __init__(self, opt):
        super(VisualEncoder, self).__init__()
        # embedding (input) layer options
        self.feat_size = opt.feat_size
        self.embed_dim = opt.word_embed_dim
        # rnn layer options
        self.rnn_type = opt.rnn_type
        self.num_layers = opt.num_layers
        self.hidden_dim = opt.hidden_dim
        self.dropout = opt.visual_dropout
        self.story_size = opt.story_size
        self.with_position = opt.with_position
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

        # visual embedding layer
        self.visual_emb = nn.Linear(self.feat_size, self.embed_dim)
        # self.visual_emb = nn.Sequential(nn.Linear(self.feat_size, self.embed_dim),
        #                                 nn.BatchNorm1d(self.embed_dim),
        #                                 nn.ReLU(True))

        # visual rnn layer
        self.hin_dropout_layer = nn.Dropout(self.dropout)
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=self.embed_dim, hidden_size=self.hidden_dim // 2,
                              dropout=self.dropout, batch_first=True, bidirectional=True)
            self.rnn_tell = nn.GRU(input_size=self.embed_dim, hidden_size=self.hidden_dim // 2, dropout=self.dropout, batch_first=True, bidirectional=True)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_dim // 2,dropout=self.dropout, batch_first=True, bidirectional=True)
            self.rnn_tell = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_dim // 2,dropout=self.dropout, batch_first=True, bidirectional=True)
        else:
            raise Exception("RNN type is not supported: {}".format(self.rnn_type))

        # residual part
        self.project_layer = nn.Linear(self.hidden_dim, self.embed_dim)
        self.relu = nn.ReLU()

        if self.with_position:
            self.position_embed = nn.Embedding(self.story_size, self.embed_dim)

        # self.z1 = nn.Conv1d(512,256,1)
        # self.z2 = nn.Conv1d(512,256,1)
        # self.g = nn.Conv1d(512,256,1)
        # self.l = nn.Conv1d(256,512,1)
        # self.z1_tell = nn.Conv1d(512,256,1)
        # self.z2_tell = nn.Conv1d(512,256,1)
        # self.g_tell = nn.Conv1d(512,256,1)
        # self.l_tell = nn.Conv1d(256,512,1)

    def init_hidden(self, batch_size, bi, dim):
        # the first parameter from the class
        weight = next(self.parameters()).data
        times = 2 if bi else 1
        if self.rnn_type == 'gru':
            return weight.new(self.num_layers * times, batch_size, dim).zero_()
        else:
            return (weight.new(self.num_layers * times, batch_size, dim).zero_(),
                    weight.new(self.num_layers * times, batch_size, dim).zero_())

    def forward(self, input, hidden=None):
        """
        inputs:
        - input  	(batch_size, 5, feat_size)
        - hidden 	(num_layers * num_dirs, batch_size, hidden_dim // 2)
        return:
        - out 		(batch_size, 5, rnn_size), serve as context
        """
        batch_size, seq_length = input.size(0), input.size(1)

        # visual embeded
        emb = self.visual_emb(input.view(-1, self.feat_size))
        emb = emb.view(batch_size, seq_length, -1)  # (Na, album_size, embedding_size)

        # visual rnn layer
        # if hidden is None:
        #     hidden = self.init_hidden(batch_size, bi=True, dim=self.hidden_dim // 2)
        # rnn_input = self.hin_dropout_layer(emb)  # apply dropout
        # # print(rnn_input.size(), hidden.size())
        # houts, hidden = self.rnn(rnn_input, hidden)

        # out = houts 
        out = emb
        # if self.with_position:
        #     for i in range(self.story_size):
        #         position = input.data.new(batch_size).long().fill_(i)
        #         out[:, i, :] = out[:, i, :] + self.position_embed(position)
        # out = out + emb
        # out = torch.selu(out)
        # out = self.layer_norm(out)
        # out = torch.selu(out)
        # out = self.layer_norm(out)
        # print("out", out.size())
        
        # # out = self.dropout1(out)
        # out1 = self.z1(out).transpose(1,2)
        # out2 = self.z2(out)
        # out_mat = torch.bmm(out1, out2)
        # out_mat = torch.softmax(out_mat, dim=-1)
        # out = self.g(out)
        # out = torch.bmm(out_mat, out.transpose(1,2))
        # out = self.l(out.transpose(1,2)).transpose(1,2)+houts
        # # residual layer
        # out = emb + out
        # out = torch.selu(out)  # (batch_size, 5, embed_dim)

        # # out = self.dropout2(out)
        # hidden = self.init_hidden(batch_size, bi=True, dim=self.hidden_dim // 2)
        # rnn_input = self.hin_dropout_layer(out)  # apply dropout
        # houts, hidden = self.rnn_tell(rnn_input, hidden)
        # # out = self.dropout3(out)
        # out = houts.transpose(1,2)
        # # out = self.dropout1(out)
        # out1 = self.z1_tell(out).transpose(1,2)
        # out2 = self.z2_tell(out)
        # out_mat = torch.bmm(out1, out2)
        # out_mat = torch.softmax(out_mat, dim=-1)
        # out = self.g_tell(out)
        # out = torch.bmm(out_mat, out.transpose(1,2))
        # out = self.l_tell(out.transpose(1,2)).transpose(1,2)+houts
        # # residual layer
        # # out = emb + outq
        # out = torch.selu(out)  # (batch_size, 5, embed_dim)

        return out



class BARTModelWrapper(nn.Module):
    def __init__(self, opt):
        nn.Module.__init__(self)

        self._device_encoder = self._device_decoder = None
        
        self._interface = torch.hub.load('pytorch/fairseq', 'bart.large')

        self._n_split_gpus = None

        self.visual_encoder = VisualEncoder(opt)

    def split_to_gpus(self, n_gpus):
        if self._n_split_gpus == n_gpus:
            return

        assert n_gpus <= 2
        assert n_gpus <= torch.cuda.device_count()

        if n_gpus == 2:
            self.encoder.cuda(0)
            self.decoder.cuda(1)
            self.visual_encoder.cuda(0)
            self._device_encoder = 'cuda:0'
            self._device_decoder = 'cuda:1'
        else:
            self.cuda()
            self._device_encoder = self._device_decoder = 'cuda'

        torch.cuda.empty_cache()
        self._n_split_gpus = n_gpus

    def forward(self, src_feat, keys, src_lengths, prev_output_tokens):
        encoder_out = forward_encoder(
            self=self.encoder,
            src_feat=src_feat,
            keys = keys,
            src_lengths=torch.tensor([src_feat.shape[1]+keys.shape[1]]),
            device_encoder=self.visual_encoder)

        for key in encoder_out:
            if isinstance(encoder_out[key], torch.Tensor):
                encoder_out[key] = encoder_out[key].to(self._device_decoder)

        x, extra = forward_decoder(
            self=self.decoder,
            device_decoder=self._device_decoder,
            prev_output_tokens=prev_output_tokens.to(self._device_decoder),
            encoder_out=encoder_out,
            features_only=False)

        return x, extra

    @property
    def model(self):
        return self._interface.model

    @property
    def encode(self):
        return self._interface.encode

    @property
    def decode(self):
        return self._interface.decode

    @property
    def encoder(self):
        return self._interface.model.encoder

    @property
    def decoder(self):
        return self._interface.model.decoder

    @property
    def dictionary(self):
        return self._interface.model.decoder.dictionary


def forward_embedding(self, src_tokens, device_embed_tokens, device_encoder):
    # embed tokens and positions
    embed = self.embed_scale * self.embed_tokens(
        src_tokens.to(device_embed_tokens))

    if self.embed_positions is not None:
        x = embed + self.embed_positions(src_tokens.to('cuda'))
    if self.layernorm_embedding:
        x = self.layernorm_embedding(x)
    # x = F.dropout(x, p=self.dropout, training=self.training)
    return x, embed

def forward_encoder(self, src_feat, keys, src_lengths, device_encoder, cls_input=None,
            return_all_hiddens=False, **unused):
    """
    Args:
        src_tokens (LongTensor): tokens in the source language of shape
            `(batch, src_len)`
        src_lengths (torch.LongTensor): lengths of each source sentence of
            shape `(batch)`
        return_all_hiddens (bool, optional): also return all of the
            intermediate hidden states (default: False).
    Returns:
        dict:
            - **encoder_out** (Tensor): the last encoder layer's output of
              shape `(src_len, batch, embed_dim)`
            - **encoder_padding_mask** (ByteTensor): the positions of
              padding elements of shape `(batch, src_len)`
            - **encoder_states** (List[Tensor]): all intermediate
              hidden states of shape `(src_len, batch, embed_dim)`.
              Only populated if *return_all_hiddens* is True.
    """
    # if self.layer_wise_attention:
    #     return_all_hiddens = True
    return_all_hiddens = True


    src_feat = device_encoder(src_feat)
    # B x T x C -> T x B x C
    key, encoder_embedding = forward_embedding(
        self=self,
        src_tokens=keys,
        device_embed_tokens=self.embed_tokens.weight.device,
        device_encoder=device_encoder)

    x = torch.cat([src_feat, key], dim=-2)
    # print(src_feat.size(), key.size())

    x = x.transpose(0, 1)

    encoder_padding_mask = None

    encoder_states = [] if return_all_hiddens else None

    # encoder layers !!!
    for layer in self.layers:
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        dropout_probability = random.uniform(0, 1)
        if not self.training or (dropout_probability > self.encoder_layerdrop):
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                encoder_states.append(x)

    if self.layer_norm:
        x = self.layer_norm(x)
        if return_all_hiddens:
            encoder_states[-1] = x

    return {
        'encoder_out': x,  # T x B x C
        'encoder_padding_mask': encoder_padding_mask,  # B x T
        'encoder_embedding': encoder_embedding,  # B x T x C
        'encoder_states': encoder_states,  # List[T x B x C]
    }


def forward_decoder(
        self,
        prev_output_tokens,
        device_decoder,
        encoder_out=None,
        incremental_state=None,
        features_only=False,
        **extra_args,
):
    """
    Args:
        prev_output_tokens (LongTensor): previous decoder outputs of shape
            `(batch, tgt_len)`, for teacher forcing
        encoder_out (Tensor, optional): output from the encoder, used for
            encoder-side attention
        incremental_state (dict): dictionary used for storing state during
            :ref:`Incremental decoding`
        features_only (bool, optional): only return features without
            applying output layer (default: False).
    Returns:
        tuple:
            - the decoder's output of shape `(batch, tgt_len, vocab)`
            - a dictionary with any model-specific outputs
    """

    x, extra = extract_features(
        self=self,
        device_embed_tokens=self.embed_tokens.weight.device,
        device_decoder=device_decoder,
        prev_output_tokens=prev_output_tokens,
        encoder_out=encoder_out,
        incremental_state=incremental_state,
        **extra_args)
    # print(prev_output_tokens)
    if not features_only:
        if self.share_input_output_embed:
            x = x.to(self.embed_tokens.weight.device)
        x = self.output_layer(x)
    return x, extra


def extract_features(
    self,
    prev_output_tokens,
    device_embed_tokens,
    device_decoder,
    encoder_out=None,
    incremental_state=None,
    full_context_alignment=False,
    alignment_layer=None,
    alignment_heads=None,
    **unused,
):
    if alignment_layer is None:
        alignment_layer = len(self.layers) - 1

    # embed positions
    positions = self.embed_positions(
        prev_output_tokens,
        incremental_state=incremental_state,
    ) if self.embed_positions is not None else None

    if incremental_state is not None:
        prev_output_tokens = prev_output_tokens[:, -1:]
        if positions is not None:
            positions = positions[:, -1:]

    # embed tokens and positions
    prev_output_tokens_embedding = self.embed_tokens(
        prev_output_tokens.to(device_embed_tokens)).to(device_decoder)

    x = self.embed_scale * prev_output_tokens_embedding

    if self.project_in_dim is not None:
        x = self.project_in_dim(x)

    if positions is not None:
        x += positions

    if self.layernorm_embedding:
        x = self.layernorm_embedding(x)

    # x = F.dropout(x, p=self.dropout, training=self.training)

    # B x T x C -> T x B x C
    x = x.transpose(0, 1)

    self_attn_padding_mask = None
    if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
        self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

    # decoder layers
    attn = None
    inner_states = [x]
    for idx, layer in enumerate(self.layers):
        encoder_state = None
        if encoder_out is not None:
            # if self.layer_wise_attention:
            #     encoder_state = encoder_out['encoder_states'][idx]
            # else:
            #     encoder_state = encoder_out['encoder_out']
            encoder_state = encoder_out['encoder_out']

        if incremental_state is None and not full_context_alignment:
            self_attn_mask = self.buffered_future_mask(x)
        else:
            self_attn_mask = None
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        # print(type(layer))
        dropout_probability = random.uniform(0, 1)
        if not self.training or (dropout_probability > self.decoder_layerdrop):
            # x, layer_attn = layer(
            #     x,
            #     encoder_state,
            #     encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
            #     incremental_state,
            #     self_attn_mask=self_attn_mask,
            #     self_attn_padding_mask=self_attn_padding_mask,
            #     need_attn=(idx == alignment_layer),
            #     need_head_weights=(idx == alignment_layer),
            # )
            x, layer_attn, _ = layer(
                x,
                encoder_state,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=(idx == alignment_layer),
                need_head_weights=(idx == alignment_layer),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float()

    if attn is not None:
        if alignment_heads is not None:
            attn = attn[:alignment_heads]

        # average probabilities over heads
        attn = attn.mean(dim=0)

    if self.layer_norm:
        x = self.layer_norm(x)

    # T x B x C -> B x T x C
    x = x.transpose(0, 1)

    if self.project_out_dim is not None:
        x = self.project_out_dim(x)

    return x, {'attn': attn, 'inner_states': inner_states}
