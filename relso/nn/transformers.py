
import math
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule

import argparse
import wandb


from relso.nn.bneck import BaseBottleneck
from relso.nn.pooling import AttnPooling

from relso.nn.base import BaseModel, BaseModelVAE, BaseVAEParamModule
from relso.nn.convolutional import Block
from relso.nn.auxnetwork import str2auxnetwork


# ---------------------
# Tranformer Block
# ---------------------

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # regiaster_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


        
class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()


    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)


    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o



class EncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])


    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x


    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps




        
# --------------------
# ReLSO model
# --------------------

class Relso(BaseModel):
    def __init__(self, hparams):
        super(Relso, self).__init__(hparams)

        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        self.save_hyperparameters()
        self.hparams = hparams

        #model atributes
        self.input_dim = hparams.input_dim
        self.latent_dim = hparams.latent_dim
        self.hidden_dim = hparams.hidden_dim # nhid
        self.embedding_dim = hparams.embedding_dim
        self.kernel_size = hparams.kernel_size
        self.layers = hparams.layers
        self.probs = hparams.probs
        self.nhead = 4
        self.src_mask = None
        self.bz = hparams.batch_size

        self.lr = hparams.lr
        self.opt_method = hparams.opt_method
        self.model_name = hparams.model


        if hparams.track_grads:
            self.bw_align_sum = 0
            self.bw_align_sum_count = 0

        if hparams.opt_method == 'gradnorm':

            self.alpha = torch.tensor(hparams.alpha_val, requires_grad=True)
            self.gamma = torch.tensor(hparams.gamma_val, requires_grad=True)

            # initialize the initial loss attributes
            self.l_recon_0 = None
            self.l_reg_0 = None

        else:
            self.alpha = hparams.alpha_val
            self.gamma = hparams.gamma_val

        self.sigma = hparams.sigma_val

        try:
            self.eta = hparams.eta_val
        except:
            self.eta = 1.0


        
        try:
            self.interp_samping = hparams.interp_samp
        except:
            self.interp_samping = True
        
        try:
            self.negative_sampling = hparams.neg_samp
        except:
            self.negative_sampling = True


        try:
            self.neg_size = hparams.neg_size
            self.neg_floor = hparams.neg_floor
            self.neg_weight = hparams.neg_weight
            self.neg_focus = hparams.neg_focus
            self.neg_norm = hparams.neg_norm
        except:
            self.neg_focus = False
            

        self.dyn_neg_bool = False  # set False as default

        try:
            self.interp_size = hparams.interp_size
            self.interp_weight = hparams.interp_weight

        except:
            pass
        self.interp_inds = None
        self.dyn_interp_bool = False

        try: 
            self.wl2norm = hparams.wl2norm
        except:
            self.wl2norm = False


        self.g_opt_step = 0


        self.seq_len = hparams.seq_len

        # The embedding input dimension may need to be changed depending on the size of our dictionary
        self.embed = nn.Embedding(self.input_dim,
                                 self.embedding_dim)

        self.pos_encoder = PositionalEncoding(d_model=self.embedding_dim,
                                            max_len = self.seq_len)

        self.glob_attn_module = nn.Sequential(nn.Linear(self.embedding_dim, 1),
                                            nn.Softmax(dim=1))


        self.transformer_encoder = TransformerEncoder(num_layers=self.layers,
                                                    input_dim=self.embedding_dim,
                                                    num_heads= self.nhead,
                                                    dim_feedforward=self.hidden_dim,
                                                    dropout=self.probs)
        # make decoder)
        self._build_decoder(hparams)

        # for los and gradient checking
        self.z_rep = None

        # auxiliary network
        self.bottleneck_module = BaseBottleneck(self.embedding_dim, self.latent_dim)

        # self.bottleneck = BaseBottleneck(self.embedding_dim, self.latent_dim)
        aux_params = {'latent_dim': self.latent_dim, 'probs': hparams.probs}
        aux_hparams = argparse.Namespace(**aux_params)

        try:
            auxnetwork = str2auxnetwork(hparams.auxnetwork)
            print(auxnetwork)
            self.regressor_module = auxnetwork(aux_hparams)
        except:
            auxnetwork = str2auxnetwork('base_reg')
            self.regressor_module = auxnetwork(aux_hparams)

    def _generate_square_subsequent_mask(self, sz):
        """create mask for transformer
        Args:
            sz (int): sequence length
        Returns:
            torch tensor : returns upper tri tensor of shape (S x S )
        """
        mask = torch.ones((sz,sz), device=self.device)
        mask = (torch.triu(mask) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _build_decoder(self, hparams):
        dec_layers = [nn.Linear(self.latent_dim, self.seq_len*(self.hidden_dim//2)),
                     Block(self.hidden_dim//2, self.hidden_dim),
                     nn.Conv1d(self.hidden_dim, self.input_dim, kernel_size=3, padding=1)]
        self.dec_conv_module = nn.ModuleList(dec_layers)

        print(dec_layers)


    def transformer_encoding(self, embedded_batch):
        """transformer logic
        Args:
            embedded_batch ([type]): output of nn.Embedding layer (B x S X E)
        mask shape: S x S
        """
        if self.src_mask is None or self.src_mask.size(0) != len(embedded_batch):
            self.src_mask = self._generate_square_subsequent_mask(embedded_batch.size(1))
        # self.embed gives output (batch_size,sequence_length,num_features)
        pos_encoded_batch = self.pos_encoder(embedded_batch)

        # TransformerEncoder takes input (sequence_length,batch_size,num_features)
        output_embed =  self.transformer_encoder(pos_encoded_batch, self.src_mask)

        return output_embed

    def encode(self,batch):
        embedded_batch = self.embed(batch)
        output_embed = self.transformer_encoding(embedded_batch)

        glob_attn = self.glob_attn_module(output_embed) # output should be B x S x 1
        z_rep = torch.bmm(glob_attn.transpose(-1, 1), output_embed).squeeze()

        # to regain the batch dimension
        if len(embedded_batch) == 1:
            z_rep = z_rep.unsqueeze(0)

        z_rep = self.bottleneck_module(z_rep)
        
        return z_rep


    def decode(self, z_rep):

        h_rep = z_rep # B x 1 X L

        for indx, layer in enumerate(self.dec_conv_module):
            if indx == 1:
                h_rep = h_rep.reshape(-1, self.hidden_dim//2, self.seq_len)

            h_rep = layer(h_rep)

        return h_rep

    def interpolation_sampling(self, z_rep):
        """
        get interpolations between z_reps in batch 
        interpolations between z_i and its nearest neighbor
        Args:
            z_rep ([type]): [description]
        Returns:
            [type]: [description]
        """
        z_dist_mat = self.pairwise_l2(z_rep, z_rep)
        k_val = min(len(z_rep), 2)
        _ ,z_nn_inds = z_dist_mat.topk(k_val, largest=False)
        z_nn_inds = z_nn_inds[:,1]

        z_nn = z_rep[z_nn_inds]
        z_interp = (z_rep + z_nn)/2

        # use 10 % of batch size
        subset_inds = torch.randperm(len(z_rep), device=self.device)[:self.interp_size]

        sub_z_interp = z_interp[subset_inds]
        sub_nn_inds = z_nn_inds[subset_inds]

        self.interp_inds = torch.cat( (subset_inds.unsqueeze(1),
                                  sub_nn_inds.unsqueeze(1)),
                                 dim=1)

        return sub_z_interp


    def add_negative_samples(self, ):
            
            max2norm = torch.norm(self.z_rep , p=2,dim=1).max()
            wandb.log({'max2norm':max2norm})
            rand_inds  = torch.randperm(len(self.z_rep))
            if self.neg_focus:
                neg_z = 0.5*torch.randn_like(self.z_rep)[:self.neg_size] + self.z_rep[rand_inds][:self.neg_size]
                neg_z = neg_z / torch.norm(neg_z, 2,dim=1).reshape(-1,1) 
                neg_z = neg_z * (max2norm * self.neg_norm)

            else: 
                center = self.z_rep.mean(0, keepdims=True)
                dist_set = self.z_rep - center

                # gets maximally distant rep from center
                dist_sort = torch.norm(dist_set, 2,dim=1).reshape(-1,1).sort().indices[-1]
                max_dist = dist_set[dist_sort]
                adj_dist = self.neg_norm * max_dist.repeat(len(self.z_rep),1) - dist_set
                neg_z = self.z_rep + adj_dist
                neg_z = neg_z[rand_inds][:self.neg_size]
                
            #else:
            #neg_z = torch.randn_like(self.z_rep)[:self.neg_size]
           
            return  neg_z

    def forward(self, batch):

        z_rep = self.encode(batch)
        self.z_rep = z_rep 

    
        # interpolative samping
        # ---------------------------------------
        # only do interpolative sampling if batch size is expected size
        self.dyn_interp_bool = self.interp_samping and z_rep.size(0) == self.bz
        if self.dyn_interp_bool:
            z_i_rep = self.interpolation_sampling(z_rep)
            interp_z_rep = torch.cat((z_rep, z_i_rep), 0)

            x_hat = self.decode(interp_z_rep)

        else:
            
            x_hat = self.decode(z_rep)

        # negative sampling 
        # ---------------------------------------
        self.dyn_neg_bool = self.negative_sampling and z_rep.size(0) == self.bz
        if self.dyn_neg_bool:
            z_n_rep = self.add_negative_samples()
            neg_z_rep = torch.cat((z_rep, z_n_rep), 0)

            y_hat = self.regressor_module(neg_z_rep)
        else:
            y_hat = self.regressor_module(z_rep)

        # safety precaution: not sure if I can
        # overwrite variables used in autograd tape
      
        return [x_hat, y_hat], z_rep



    def pairwise_l2(self, x,y):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        
        dist = torch.pow(x - y, 2).sum(2) 

        return dist
                


    def loss_function(self, predictions, targets, alpha, gamma, valid_step=False):

        # unpack everything
        x_hat, y_hat = predictions
        x_true, y_true = targets
        

        if self.dyn_interp_bool:
            recon_x = x_hat[:-self.interp_size]
        else:
            recon_x = x_hat

        # lower weight of padding token in loss
        ce_loss_weights = torch.ones(22, device=self.device)
        ce_loss_weights[21] = 0.8

        ae_loss = nn.CrossEntropyLoss(weight=ce_loss_weights)(recon_x, x_true)
        ae_loss = gamma * ae_loss


        if self.dyn_neg_bool:
            pred_y = y_hat[:-self.neg_size]
            extend_y = y_hat[-self.neg_size:]
        else:
            pred_y = y_hat

        # enrichment pred loss
        reg_loss = nn.MSELoss()(pred_y.flatten(), y_true.flatten())
        reg_loss = alpha * reg_loss


        # interpolation loss
        # z_dist_mat = self.pairwise_l2(self.z_rep, self.z_rep)
        if self.dyn_interp_bool:
            seq_preds = F.gumbel_softmax(x_hat, tau=1, dim=1, hard=True).transpose(1,2).flatten(1,2)
            seq_dist_mat = torch.cdist(seq_preds,seq_preds, p=1)

            #rint(f' bs {self.bz}\t z_rep: {self.z_rep.size(0)}')

            ext_inds = torch.arange(self.bz, self.bz + self.interp_size)
            tr_dists = seq_dist_mat[self.interp_inds[:,0], self.interp_inds[:,1]]
            inter_dist1 = seq_dist_mat[ext_inds, self.interp_inds[:,0]]
            inter_dist2 = seq_dist_mat[ext_inds, self.interp_inds[:,1]]

            interp_loss = (0.5 * (inter_dist1 + inter_dist2) - 0.5 * tr_dists).mean()
            interp_loss = max(0,interp_loss) * self.interp_weight
        else:
            interp_loss = 0


        # negative sampling loss
        if self.dyn_neg_bool:
            neg_targets = torch.ones((self.neg_size), device=self.device) * self.neg_floor
            neg_loss = nn.MSELoss()(extend_y.flatten(), neg_targets.flatten())
            neg_loss = neg_loss * self.neg_weight
        else:
            neg_loss = 0
        
        # RAE L_z loss
        # only penalize real zs
        zrep_l2_loss = 0.5 * torch.norm(self.z_rep, 2, dim=1) ** 2
        
        if self.wl2norm:
            y_true_shift = y_true + torch.abs(y_true.min())
            w_fit_zrep = nn.ReLU()(y_true_shift / y_true_shift.sum() )
            zrep_l2_loss = torch.dot(zrep_l2_loss.flatten(), w_fit_zrep.flatten())
        else:
            zrep_l2_loss = zrep_l2_loss.mean()

        zrep_l2_loss = zrep_l2_loss * self.eta

        total_loss = ae_loss + reg_loss + zrep_l2_loss + interp_loss + neg_loss

        mloss_dict = {"ae_loss": ae_loss,
                    "zrep_l2_loss": zrep_l2_loss,
                    "interp_loss": interp_loss,
                    "neg samp loss": neg_loss, 
                    "reg_loss": reg_loss,
                    "loss": total_loss}



        return total_loss, mloss_dict
