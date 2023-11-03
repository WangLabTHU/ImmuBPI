import torch
import torch.nn as nn
from typing import Optional

from models.positionalencoding import PositionalEncoding

class Transformer_fusion(nn.Module):
    '''Transformer For Predicting immunogenicity given peptide and hla'''

    def __init__(
        self,
        vocab_size: int = 22, 
        d_model: int = 64,
        dim_feedfoward: int = 2048,
        n_layers: int = 1,
        n_head: int = 8,
        dropout: float = 0.1,
        add_cls: bool = False,
        use_sin_pos: bool = True,
    ) -> None:

        super().__init__()
        self.src_word_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout, is_sin=use_sin_pos)
        self.add_cls = add_cls
       

        self.combine_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_head, dim_feedfoward, dropout, batch_first=True, norm_first=True),
            n_layers
        )

        self.cls_token = nn.Parameter(torch.randn(1, d_model), requires_grad=True)
        self.register_buffer('cls_mask', torch.zeros((1,), dtype=bool))

        if self.add_cls:
            self.projection = nn.Sequential(
                                            nn.Linear(d_model, 64),
                                            nn.ReLU(True),
                                            nn.Linear(64, 2)
            )
        else:
            self.projection = nn.Sequential(
                                            nn.Linear(d_model * 49, 64),
                                            nn.ReLU(True),
                                            nn.Linear(64, 2)
            )        
                                        
        


    def forward(
        self, 
        epitope: torch.Tensor, 
        hla: torch.Tensor, 
        epitope_padding_mask: Optional[torch.Tensor]=None,
        hla_padding_mask: Optional[torch.Tensor]=None,
        **kwargs
    ):
        """ 
        forward loop of interactive transformer between epitope and hla

        Args:
            epitope (torch.Tensor): in shape [B, L] long tensor of epitope aa indices
            hla (torch.Tensor): in shape [B, L] long tensor of hla aa indices
        """
        B, L_e = epitope.shape
        L_h = hla.shape[1]
        epitope_emb = self.pos_enc(self.src_word_emb(epitope)) 
        hla_emb = self.pos_enc(self.src_word_emb(hla)) 

        if self.add_cls:
            combine_emb = torch.cat((self.cls_token.unsqueeze(0).repeat(epitope_emb.size(0),1,1), epitope_emb, hla_emb), 1) 
            combine_padding_mask = torch.cat((self.cls_mask.unsqueeze(0).repeat(epitope_emb.size(0),1), epitope_padding_mask, hla_padding_mask), 1) 
        else:
            combine_emb = torch.cat((epitope_emb, hla_emb), 1) 
            combine_padding_mask = torch.cat((epitope_padding_mask, hla_padding_mask), 1) 
        
        binding_emb = self.combine_encoder(combine_emb, mask=None, src_key_padding_mask=combine_padding_mask)  # type: torch.Tensor
        
        if self.add_cls:
            final_emb = binding_emb[:,0,:]
        else:
            final_emb = binding_emb.view(binding_emb.shape[0], -1) 

        output = self.projection(final_emb)
        

        # return output.view(-1, output.size(-1))
        return output
