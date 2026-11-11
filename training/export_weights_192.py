"""
export_weights_192.py — Exporta el checkpoint de 192 tokens a .bin.
Idéntico al original pero con MAX_SEQ_LEN=192.
"""
import os, struct
import numpy as np
import torch
from model_192 import BitNetTransformer

CHECKPOINT = "/content/drive/MyDrive/checkpoints_PLE_192/ep20_val1.382.pt"
OUT_DIR = "/content"
BIN_PATH = os.path.join(OUT_DIR, "bitnet_weights.bin")
H_PATH = os.path.join(OUT_DIR, "bitnet_weights.h")

def quantize_ternary_packed(w):
    w = w.detach().cpu().float()
    scale = w.abs().mean().clamp(min=1e-8).item()
    q = torch.clamp(torch.round(w / scale), -1, 1).to(torch.int8).flatten()
    codes = (q + 1).to(torch.uint8)
    pad = (-len(codes)) % 4
    if pad:
        codes = torch.cat([codes, torch.ones(pad, dtype=torch.uint8)])
    c = codes.numpy()
    packed = (c[0::4] | (c[1::4] << 2) | (c[2::4] << 4) | (c[3::4] << 6)).astype(np.uint8)
    return packed.tobytes(), scale

def quantize_int4(w):
    w = w.detach().cpu().float()
    n_rows = w.shape[0]
    scales = np.zeros(n_rows, dtype=np.float32)
    all_q = []
    for i in range(n_rows):
        row = w[i]
        s = row.abs().max().item() / 7.0
        if s < 1e-8: s = 1e-8
        scales[i] = s
        q = torch.clamp(torch.round(row / s), -7, 7).to(torch.int8)
        all_q.append(q)
    flat = torch.cat(all_q).numpy().astype(np.int8)
    u = (flat & 0x0F).astype(np.uint8)
    if len(u) % 2 != 0:
        u = np.append(u, 0)
    packed = ((u[::2] & 0x0F) | ((u[1::2] & 0x0F) << 4)).tobytes()
    return packed, scales.tobytes()

def fp16_bytes(t):
    return t.detach().cpu().float().half().numpy().tobytes()

def main():
    print("Cargando checkpoint 192 tokens...")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    print(f"Config: {cfg}")
    model = BitNetTransformer(**cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    V = cfg["vocab_size"]; E = cfg["embed_dim"]; H = cfg["num_heads"]
    KVH = cfg["num_kv_heads"]; L = cfg["num_layers"]; F = cfg["ff_dim"]
    S = cfg["max_seq_len"]; PLE = cfg["ple_hidden_dim"]

    buf = bytearray()
    offsets = {}
    def write(b, name):
        offsets[name] = len(buf)
        buf.extend(b)

    # token_embed INT4
    tb, ts_scales = quantize_int4(model.token_embed.weight)
    write(tb, "TOKEN_EMBED")
    write(ts_scales, "TOKEN_EMBED_SCALES")
    write(fp16_bytes(model.pos_embed.weight), "POS_EMBED")

    # PLE
    ple_w = model.ple.embed_tokens_per_layer.weight
    ple_bytes, ple_scales = quantize_int4(ple_w)
    write(ple_bytes, "PLE_TABLE")
    write(ple_scales, "PLE_SCALES")

    qb, sc = quantize_ternary_packed(model.ple.per_layer_model_projection.weight)
    write(qb, "PLE_PROJ_WEIGHT")
    write(struct.pack("<f", sc), "PLE_PROJ_SCALE")

    # Bloques
    for i, blk in enumerate(model.blocks):
        write(fp16_bytes(blk.norm1.weight), f"B{i}_NORM1_GAMMA")
        write(fp16_bytes(blk.norm1.bias),   f"B{i}_NORM1_BETA")
        for nm, proj in [("Q", blk.attn.q_proj), ("K", blk.attn.k_proj),
                         ("V", blk.attn.v_proj), ("O", blk.attn.out_proj)]:
            qb, sc = quantize_ternary_packed(proj.weight)
            write(qb, f"B{i}_{nm}_WEIGHT")
            write(struct.pack("<f", sc), f"B{i}_{nm}_SCALE")
            write(fp16_bytes(proj.bias), f"B{i}_{nm}_BIAS")
        write(fp16_bytes(blk.norm2.weight), f"B{i}_NORM2_GAMMA")
        write(fp16_bytes(blk.norm2.bias),   f"B{i}_NORM2_BETA")
        for nm, lin in [("FC1", blk.ffn.fc1), ("FC2", blk.ffn.fc2)]:
            qb, sc = quantize_ternary_packed(lin.weight)
            write(qb, f"B{i}_{nm}_WEIGHT")
            write(struct.pack("<f", sc), f"B{i}_{nm}_SCALE")
            write(fp16_bytes(lin.bias), f"B{i}_{nm}_BIAS")
        qb, sc = quantize_ternary_packed(blk.ple_to_residual.weight)
        write(qb, f"B{i}_PLE_RES_WEIGHT")
        write(struct.pack("<f", sc), f"B{i}_PLE_RES_SCALE")
        write(fp16_bytes(blk.ple_norm.weight), f"B{i}_PLE_NORM_GAMMA")
        write(fp16_bytes(blk.ple_norm.bias),   f"B{i}_PLE_NORM_BETA")

    write(fp16_bytes(model.norm_final.weight), "FINAL_NORM_GAMMA")
    write(fp16_bytes(model.norm_final.bias),   "FINAL_NORM_BETA")
    qb, sc = quantize_ternary_packed(model.lm_head.weight)
    write(qb, "LM_HEAD_WEIGHT")
    write(struct.pack("<f", sc), "LM_HEAD_SCALE")
    write(fp16_bytes(model.lm_head.bias), "LM_HEAD_BIAS")

    with open(BIN_PATH, "wb") as f:
        f.write(buf)
    print(f"\nBin: {BIN_PATH} ({len(buf):,} bytes = {len(buf)/1024/1024:.2f} MB)")

    h = f"""// Auto-generated.
#ifndef BITNET_WEIGHTS_H
#define BITNET_WEIGHTS_H
#include <stdint.h>
#include <stddef.h>

#define VOCAB_SIZE      {V}
#define EMBED_DIM       {E}
#define NUM_HEADS       {H}
#define NUM_KV_HEADS    {KVH}
#define NUM_LAYERS      {L}
#define FF_DIM          {F}
#define MAX_SEQ_LEN     {S}
#define HEAD_DIM        {E // H}
#define KV_DIM          {KVH * (E // H)}
#define PLE_HIDDEN_DIM  {PLE}
#define PLE_ROW_VALUES  {L * PLE}
#define PLE_ROW_BYTES   {(L * PLE) // 2}
#define TOKEN_EMBED_ROW_BYTES {E // 2}

#define WEIGHTS_BIN_SIZE {len(buf)}

"""
    h += "\n".join(f"#define OFF_{k} {v}" for k, v in offsets.items())
    h += "\n\n#endif\n"
    with open(H_PATH, "w") as f:
        f.write(h)
    print(f"Header: {H_PATH} ({os.path.getsize(H_PATH)} bytes)")
    print(f"\nTotal Flash: {len(buf)/1024/1024:.2f} MB / 4 MB")

if __name__ == "__main__":
    main()