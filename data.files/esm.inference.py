import pandas as pd
import numpy as np
import os
import esm
import torch
import argparse

os.environ['CUDA_LAUNCH_BLOCKING'] = '-1'


def precompute_sequence(transcript_id, sequence, esm_model, batch_converter, out_dir, device_id=0):
    if os.path.exists(os.path.join(out_dir, transcript_id + '.contacts.npy')):
        return
    else:
        print('begin precompute sequence for {}'.format(transcript_id))
    try:
        data = [(transcript_id, sequence)]
        _, _, toks = batch_converter(data)
    except:
        print(transcript_id)
        return
    toks = toks.to(f'cuda:{device_id}')
    aa = toks.shape[1]
    if aa <= 2250:
        print(f"{transcript_id} has {toks.shape[1]} amino acids")
        return
    with torch.no_grad():
        out = esm_model(toks, repr_layers=[33], return_contacts=True, need_head_weights=False)
    representations = out["representations"][33][0].to(device='cpu').detach().numpy()
    # output is batch x layers x heads x seqlen x seqlen
    # attentions = out["attentions"][0].to(device="cpu").detach().numpy()
    contacts = out['contacts'][0].to(device="cpu").detach().numpy()
    logits = out['logits'][0].to(device="cpu").detach().numpy()
    np.save(
        f"{out_dir}/{transcript_id}.representations.layer.48.npy",
        representations,
    )
    np.save(
        f"{out_dir}/{transcript_id}.contacts.npy",
        contacts,
    )
    np.save(
        f"{out_dir}/{transcript_id}.logits.npy",
        logits,
    )
    return


def precompute_sequence_multiple_gpus(transcript_id, sequence, esm_model, batch_converter, out_dir):
    if os.path.exists(os.path.join(out_dir, transcript_id + '.contacts.npy')):
        return
    else:
        print('begin precompute sequence for {}'.format(transcript_id))
    try:
        data = [(transcript_id, sequence)]
        _, _, toks = batch_converter(data)
    except:
        print(transcript_id)
        return
    toks = toks.to('cuda:0')
    if toks.shape[1] > 30000:
        print(f"{transcript_id} has {toks.shape[1]} amino acids, don't proceed")
        return
    print(f"{transcript_id} has {toks.shape[1]} amino acids")
    if toks.shape[1] > 5500:
        need_head_weights = False
        return_contacts = False
    else:
        need_head_weights = True
        return_contacts = True
    with torch.no_grad():
        assert toks.ndim == 2
        padding_mask = toks.eq(esm_model.padding_idx)  # B, T
        x = esm_model.embed_scale * esm_model.embed_tokens(toks)

        if esm_model.token_dropout:
            x.masked_fill_((toks == esm_model.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (toks == esm_model.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = {33}
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x
        if need_head_weights:
            attn_weights = []
        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)
        if not padding_mask.any():
            padding_mask = None
        for layer_idx, layer in enumerate(esm_model.layers):
            x = x.to(f'cuda:{layer_idx // 9}')
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0).cpu())
        x = esm_model.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        # lm head is on cuda:0, x is on cuda:3
        x = esm_model.lm_head(x.to('cuda:0'))
        out = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            out["attentions"] = attentions
            if return_contacts:
                contacts = esm_model.contact_head(toks, attentions)
                out["contacts"] = contacts
    representations = out["representations"][33][0].to(device='cpu').detach().numpy()
    # output is batch x layers x heads x seqlen x seqlen

    logits = out['logits'][0].to(device="cpu").detach().numpy()
    np.save(
        f"{out_dir}/{transcript_id}.representations.layer.48.npy",
        representations,
    )
    np.save(
        f"{out_dir}/{transcript_id}.logits.npy",
        logits,
    )
    if return_contacts:
        contacts = out['contacts'][0].to(device="cpu").detach().numpy()
        np.save(
            f"{out_dir}/{transcript_id}.contacts.npy",
            contacts,
        )
    return


def main(file=None, outdir=None):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    if torch.cuda.is_available():
        # manually split the model into 4 GPUs
        model.embed_tokens.to('cuda:0')
        for layer_idx, layer in enumerate(model.layers):
            layer.to(f'cuda:{layer_idx // 9}')
        model.emb_layer_norm_after.to('cuda:3')
        model.lm_head.to('cuda:0')
        model.contact_head.to('cpu')
        print("Transferred model to GPUs")
    # model = model.to(f'cuda:{rank}')
    if file is None:
        return
    files = pd.read_csv(file, index_col=0)
    os.makedirs(outdir, exist_ok=True)
    for transcript_id, sequence in zip(files['uniprotID'], files['sequence']):
        precompute_sequence_multiple_gpus(transcript_id, sequence, model,
                                          alphabet.get_batch_converter(),
                                          outdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default=None)
    parser.add_argument('--outdir', type=str, default=None)
    args = parser.parse_args()
    main(args.file, args.outdir)
