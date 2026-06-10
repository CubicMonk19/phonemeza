"""From-scratch LSTM encoder-decoder models for G2P.

One Encoder, one Decoder parameterised by context_mode in {"none", "fixed",
"attn"}, and a Seq2Seq wrapper. The three setups share the same encoder and
wrapper; only the decoder's use of source context differs.
"""

import numpy as np
import torch
import torch.nn as nn

from g2p.data import SOS_IDX, EOS_IDX


class LSTMCell(nn.Module):
    """LSTM cell implemented from scratch.

    Eight linear maps are used: W_* (input -> hidden, no bias) and
    U_* (hidden -> hidden, with bias). Each U_* bias plays the role of the
    gate bias b_* in the equations. Same pattern as the LSTM tutorial.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input -> hidden, bias-free.
        self.W_f = nn.Linear(input_size, hidden_size, bias=False)
        self.W_i = nn.Linear(input_size, hidden_size, bias=False)
        self.W_o = nn.Linear(input_size, hidden_size, bias=False)
        self.W_c = nn.Linear(input_size, hidden_size, bias=False)

        # Hidden -> hidden, the bias here is b_* in the equations.
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=True)
        self.U_i = nn.Linear(hidden_size, hidden_size, bias=True)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=True)
        self.U_c = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x, h, c):
        # x: (batch, input_size); h, c: (batch, hidden_size)
        f = torch.sigmoid(self.W_f(x) + self.U_f(h))     # forget gate   (A.1)
        i = torch.sigmoid(self.W_i(x) + self.U_i(h))     # input gate    (A.2)
        o = torch.sigmoid(self.W_o(x) + self.U_o(h))     # output gate   (A.3)
        c_tilde = torch.tanh(self.W_c(x) + self.U_c(h))  # candidate     (A.4)
        c_new = f * c + i * c_tilde                      # cell update   (A.5)
        h_new = o * torch.tanh(c_new)                    # hidden update (A.6)
        return h_new, c_new


class LSTMCellWithContext(nn.Module):
    """LSTM cell with a context vector z added to every gate via bias-free
    V_* maps (one bias-free linear map per gate). z has dimension hidden_size."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.W_f = nn.Linear(input_size, hidden_size, bias=False)
        self.W_i = nn.Linear(input_size, hidden_size, bias=False)
        self.W_o = nn.Linear(input_size, hidden_size, bias=False)
        self.W_c = nn.Linear(input_size, hidden_size, bias=False)

        self.U_f = nn.Linear(hidden_size, hidden_size, bias=True)
        self.U_i = nn.Linear(hidden_size, hidden_size, bias=True)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=True)
        self.U_c = nn.Linear(hidden_size, hidden_size, bias=True)

        # Context maps, bias-free (the gate bias lives on U_*).
        self.V_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V_c = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h, c, z):
        f = torch.sigmoid(self.W_f(x) + self.U_f(h) + self.V_f(z))      # A.11/A.20
        i = torch.sigmoid(self.W_i(x) + self.U_i(h) + self.V_i(z))      # A.12/A.21
        o = torch.sigmoid(self.W_o(x) + self.U_o(h) + self.V_o(z))      # A.13/A.22
        c_tilde = torch.tanh(self.W_c(x) + self.U_c(h) + self.V_c(z))   # A.14/A.23
        c_new = f * c + i * c_tilde                                     # A.15/A.24
        h_new = o * torch.tanh(c_new)                                   # A.16/A.25
        return h_new, c_new


class Encoder(nn.Module):
    """Embeds the source and runs stacked from-scratch LSTM cells (A.7)."""

    def __init__(self, src_vocab_size: int, embed_dim: int, hidden_size: int,
                 num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        # Layer 0 takes the embedding; deeper layers take the hidden state.
        self.layers = nn.ModuleList([
            LSTMCell(embed_dim if l == 0 else hidden_size, hidden_size)
            for l in range(num_layers)
        ])

    def forward(self, src):
        # src: (batch, src_len) -> all_h: (src_len, batch, hidden_size),
        #                          (h_n, c_n): (num_layers, batch, hidden_size)
        batch, src_len = src.shape
        emb = self.embedding(src)  # (batch, src_len, embed_dim)
        h = [torch.zeros(batch, self.hidden_size, device=src.device)
             for _ in range(self.num_layers)]
        c = [torch.zeros(batch, self.hidden_size, device=src.device)
             for _ in range(self.num_layers)]

        all_h = []
        for t in range(src_len):
            x = emb[:, t, :]
            for l, cell in enumerate(self.layers):
                h[l], c[l] = cell(x, h[l], c[l])
                x = h[l]                       # output feeds the next layer
            all_h.append(h[-1])                # top-layer hidden each step

        all_h = torch.stack(all_h, dim=0)      # (src_len, batch, hidden_size)
        h_n = torch.stack(h, dim=0)            # (num_layers, batch, hidden)
        c_n = torch.stack(c, dim=0)
        return all_h, (h_n, c_n)


class Decoder(nn.Module):
    """Single decoder; context_mode selects Setup 1/2/3 (A.2 / A.3 / A.4)."""

    def __init__(self, tgt_vocab_size: int, embed_dim: int, hidden_size: int,
                 num_layers: int = 1, context_mode: str = "none"):
        super().__init__()
        assert context_mode in {"none", "fixed", "attn"}
        self.context_mode = context_mode
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)

        cell_cls = LSTMCell if context_mode == "none" else LSTMCellWithContext
        self.layers = nn.ModuleList([
            cell_cls(embed_dim if l == 0 else hidden_size, hidden_size)
            for l in range(num_layers)
        ])
        self.out = nn.Linear(hidden_size, tgt_vocab_size)

    def forward(self, tgt, hidden, encoder_outputs=None, src_mask=None,
                return_attn=False):
        # tgt: (batch, tgt_len); hidden: (h_0, c_0) each (L, batch, hidden)
        batch, tgt_len = tgt.shape
        emb = self.embedding(tgt)                       # (batch, tgt_len, emb)
        h0, c0 = hidden
        h = [h0[l] for l in range(self.num_layers)]
        c = [c0[l] for l in range(self.num_layers)]

        if self.context_mode == "fixed":
            z_fixed = encoder_outputs[-1]               # (batch, hidden)

        logits_steps, attn_steps = [], []
        for t in range(tgt_len):
            x = emb[:, t, :]
            if self.context_mode == "none":
                for l, cell in enumerate(self.layers):
                    h[l], c[l] = cell(x, h[l], c[l])
                    x = h[l]
            else:
                if self.context_mode == "fixed":
                    z = z_fixed
                else:  # "attn": query is previous top-layer decoder hidden
                    query = h[-1]                       # (batch, hidden)
                    scores = torch.einsum("bh,sbh->bs", query,
                                          encoder_outputs)  # (batch, src_len)
                    if src_mask is not None:
                        scores = scores.masked_fill(~src_mask, float("-inf"))
                    alpha = torch.softmax(scores, dim=1)    # (batch, src_len)
                    z = torch.einsum("bs,sbh->bh", alpha,
                                     encoder_outputs)        # (batch, hidden)
                    if return_attn:
                        attn_steps.append(alpha)
                for l, cell in enumerate(self.layers):
                    h[l], c[l] = cell(x, h[l], c[l], z)
                    x = h[l]
            logits_steps.append(self.out(h[-1]))

        logits = torch.stack(logits_steps, dim=1)       # (batch, tgt_len, V)
        h_n = torch.stack(h, dim=0)
        c_n = torch.stack(c, dim=0)
        if return_attn:
            attn = (torch.stack(attn_steps, dim=1)
                    if self.context_mode == "attn" else None)
            return logits, (h_n, c_n), attn
        return logits, (h_n, c_n)


class Seq2Seq(nn.Module):
    """Encoder-decoder wrapper shared by all three context modes."""

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing: bool = True):
        if not teacher_forcing:
            raise NotImplementedError(
                "Only teacher forcing is supported for training; "
                "use greedy_decode for inference.")
        encoder_outputs, (h_n, c_n) = self.encoder(src)
        src_mask = (src != 0)                       # (batch, src_len)
        dec_in = tgt[:, :-1]                        # drop trailing <EOS>
        logits, _ = self.decoder(dec_in, (h_n, c_n),
                                 encoder_outputs=encoder_outputs,
                                 src_mask=src_mask)
        return logits                              # (batch, tgt_len-1, vocab)

    def greedy_decode(self, src, max_len: int = 30):
        with torch.no_grad():
            encoder_outputs, (h_n, c_n) = self.encoder(src)
            src_mask = (src != 0)
            batch = src.size(0)
            cur = torch.full((batch, 1), SOS_IDX, dtype=torch.long,
                             device=src.device)
            hidden = (h_n, c_n)
            finished = torch.zeros(batch, dtype=torch.bool,
                                   device=src.device)
            preds = [[] for _ in range(batch)]
            save_attn = self.decoder.context_mode == "attn"
            attn_acc = [[] for _ in range(batch)] if save_attn else None

            for _ in range(max_len):
                if save_attn:
                    logits, hidden, attn = self.decoder(
                        cur, hidden, encoder_outputs=encoder_outputs,
                        src_mask=src_mask, return_attn=True)
                    step_attn = attn[:, 0, :]          # (batch, src_len)
                else:
                    logits, hidden = self.decoder(
                        cur, hidden, encoder_outputs=encoder_outputs,
                        src_mask=src_mask)
                next_tok = logits[:, -1, :].argmax(dim=-1)   # (batch,)
                for b in range(batch):
                    if finished[b]:
                        continue
                    tok = next_tok[b].item()
                    if tok == EOS_IDX:
                        finished[b] = True
                    else:
                        preds[b].append(tok)
                        if save_attn:
                            attn_acc[b].append(
                                step_attn[b].cpu().numpy())
                cur = next_tok.unsqueeze(1)
                if bool(finished.all()):
                    break

        if save_attn:
            src_len = src.size(1)
            attn_out = [np.stack(a, axis=0) if len(a) > 0
                        else np.zeros((0, src_len)) for a in attn_acc]
            return preds, attn_out
        return preds, None
