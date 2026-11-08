"""
export_weights.py — Exporta modelo PLE a .bin
Formato:
  - Tabla PLE: INT4 (con scale por fila) → Flash
  - Núcleo (Q,K,V,O,FC1,FC2,LM head, ple_to_residual, per_layer_proj): BitNet ternario
  - token_embed: INT8
  - pos_embed, LayerNorms, biases: FP16
"""
import os, json, struct
import numpy as np
import torch
from model import BitNetTransformer

CHECKPOINT = r"C:\Users\OPTIPLEX 5050\Desktop\IA\Modelo_01\checkpoints_PLE_TinyStories\ep20_val1.537.pt"
OUT_DIR    = r"C:\Users\OPTIPLEX 5050\Desktop\IA\Modelo_01\bitnet-to-c"
BIN_PATH   = os.path.join(OUT_DIR, "bitnet_weights.bin")
H_PATH     = os.path.join(OUT_DIR, "bitnet_weights.h")
os.makedirs(OUT_DIR, exist_ok=True)


def quantize_ternary(w):
    w = w.detach().cpu().float()
    scale = w.abs().mean().clamp(min=1e-8).item()
    q = torch.clamp(torch.round(w / scale), -1, 1).to(torch.int8)
    return q.numpy().tobytes(), scale


def quantize_int4(w):
    """Cuantiza a INT4 [-7, 7] con scale por fila. Devuelve bytes empaquetados + scales."""
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
    # Empaquetar 2 valores INT4 por byte
    u = (flat & 0x0F).astype(np.uint8)
    if len(u) % 2 != 0:
        u = np.append(u, 0)
    packed = ((u[::2] & 0x0F) | ((u[1::2] & 0x0F) << 4)).tobytes()
    return packed, scales.tobytes()


def floats_to_fp16_bytes(t):
    return t.detach().cpu().float().half().numpy().tobytes()


def int8_bytes(t):
    """Cuantiza a INT8 con scale global."""
    t = t.detach().cpu().float()
    s = t.abs().max().item() / 127.0
    if s < 1e-8: s = 1e-8
    q = torch.clamp(torch.round(t / s), -127, 127).to(torch.int8)
    return q.numpy().tobytes(), s


def main():
    print("Cargando checkpoint...")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    print(f"Config: {cfg}")
    model = BitNetTransformer(**cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    V = cfg["vocab_size"]; E = cfg["embed_dim"]; H = cfg["num_heads"]
    KVH = cfg["num_kv_heads"]; L = cfg["num_layers"]; F = cfg["ff_dim"]
    S = cfg["max_seq_len"]; PLE = cfg["ple_hidden_dim"]
    KV_DIM = KVH * (E // H)

    buf = bytearray()
    offsets = {}

    def write(b, name):
        offsets[name] = len(buf)
        buf.extend(b)

    # Embeddings (token_embed INT4 con scale por fila)
    tb, ts_scales = quantize_int4(model.token_embed.weight)
    write(tb, "TOKEN_EMBED")
    write(ts_scales, "TOKEN_EMBED_SCALES")

    # 2) pos_embed FP16
    write(floats_to_fp16_bytes(model.pos_embed.weight), "POS_EMBED")

    # 3) Tabla PLE INT4 (empaquetada) + scales por fila
    ple_w = model.ple.embed_tokens_per_layer.weight  # [V, L*PLE]
    ple_bytes, ple_scales = quantize_int4(ple_w)
    write(ple_bytes, "PLE_TABLE")
    write(ple_scales, "PLE_SCALES")  # V floats

    # 4) per_layer_model_projection (BitLinear ternario)
    qb, sc = quantize_ternary(model.ple.per_layer_model_projection.weight)
    write(qb, "PLE_PROJ_WEIGHT")
    write(struct.pack("<f", sc), "PLE_PROJ_SCALE")

    # 5) Bloques
    for i, blk in enumerate(model.blocks):
        # norm1
        write(floats_to_fp16_bytes(blk.norm1.weight), f"B{i}_NORM1_GAMMA")
        write(floats_to_fp16_bytes(blk.norm1.bias),   f"B{i}_NORM1_BETA")
        # Q/K/V/O
        for name, proj in [("Q", blk.attn.q_proj), ("K", blk.attn.k_proj),
                           ("V", blk.attn.v_proj), ("O", blk.attn.out_proj)]:
            qb, sc = quantize_ternary(proj.weight)
            write(qb, f"B{i}_{name}_WEIGHT")
            write(struct.pack("<f", sc), f"B{i}_{name}_SCALE")
            write(floats_to_fp16_bytes(proj.bias), f"B{i}_{name}_BIAS")
        # norm2
        write(floats_to_fp16_bytes(blk.norm2.weight), f"B{i}_NORM2_GAMMA")
        write(floats_to_fp16_bytes(blk.norm2.bias),   f"B{i}_NORM2_BETA")
        # FC1/FC2
        for name, lin in [("FC1", blk.ffn.fc1), ("FC2", blk.ffn.fc2)]:
            qb, sc = quantize_ternary(lin.weight)
            write(qb, f"B{i}_{name}_WEIGHT")
            write(struct.pack("<f", sc), f"B{i}_{name}_SCALE")
            write(floats_to_fp16_bytes(lin.bias), f"B{i}_{name}_BIAS")
        # ple_to_residual
        qb, sc = quantize_ternary(blk.ple_to_residual.weight)
        write(qb, f"B{i}_PLE_RES_WEIGHT")
        write(struct.pack("<f", sc), f"B{i}_PLE_RES_SCALE")
        # ple_norm
        write(floats_to_fp16_bytes(blk.ple_norm.weight), f"B{i}_PLE_NORM_GAMMA")
        write(floats_to_fp16_bytes(blk.ple_norm.bias),   f"B{i}_PLE_NORM_BETA")

    # 6) Final norm
    write(floats_to_fp16_bytes(model.norm_final.weight), "FINAL_NORM_GAMMA")
    write(floats_to_fp16_bytes(model.norm_final.bias),   "FINAL_NORM_BETA")

    # 7) LM head
    qb, sc = quantize_ternary(model.lm_head.weight)
    write(qb, "LM_HEAD_WEIGHT")
    write(struct.pack("<f", sc), "LM_HEAD_SCALE")
    write(floats_to_fp16_bytes(model.lm_head.bias), "LM_HEAD_BIAS")

    # Guardar bin
    with open(BIN_PATH, "wb") as f:
        f.write(buf)
    print(f"✅ {BIN_PATH} ({len(buf):,} bytes = {len(buf)/1024/1024:.2f} MB)")

    # Generar header
    h = f"""// Auto-generated. Do not edit.
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
#define KV_DIM          {KV_DIM}
#define PLE_HIDDEN_DIM  {PLE}
#define PLE_ROW_VALUES  {L * PLE}
#define PLE_ROW_BYTES   {(L * PLE) // 2}

#define WEIGHTS_BIN_SIZE {len(buf)}

"""
    h += "\n".join(f"#define OFF_{k} {v}" for k, v in offsets.items())
    h += "\n\n#endif\n"
    with open(H_PATH, "w") as f:
        f.write(h)
    print(f"✅ {H_PATH} ({os.path.getsize(H_PATH)} bytes)")
    print(f"\n💾 Flash total: {len(buf)/1024/1024:.2f} MB / 4 MB")


if __name__ == "__main__":
    main()
