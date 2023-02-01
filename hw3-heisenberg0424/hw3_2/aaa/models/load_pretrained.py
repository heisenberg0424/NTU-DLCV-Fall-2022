import torch
import torch.nn as nn

from models.transformer import TransformerDecoderLayer, TransformerDecoder

def load_pretrained():

    d_model=256
    nhead=8
    num_decoder_layers=6
    dim_feedforward=2048
    dropout=0.1
    activation="relu"
    normalize_before=False,
    return_intermediate_dec=False

    model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)

    for p in model.backbone.parameters():
        p.requires_grad = False

    decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
    decoder_norm = nn.LayerNorm(d_model)

    model.transformer.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)

    return model