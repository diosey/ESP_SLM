"""
train_192.py — Entrenamiento base con contexto de 192 tokens.
MAX_SEQUENCES = 300_000, 20 épocas.
"""
import os, time, math, random, gc, pickle
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model_192 import BitNetTransformer
from tokenizer import BigramTokenizer

torch.set_num_threads(2)

TOKENIZED_CORPUS = "/content/corpus_tinystories_tokenized.pkl"
OUTPUT_DIR = "/content/drive/MyDrive/checkpoints_PLE_192"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMBED_DIM = 192
NUM_HEADS = 4
NUM_KV_HEADS = 1
NUM_LAYERS = 6
FF_DIM = 384
MAX_SEQ_LEN = 192
PLE_HIDDEN_DIM = 320
DROPOUT = 0.1

BATCH_SIZE = 64
EPOCHS = 20
LR = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
VAL_SPLIT = 0.05
NUM_WORKERS = 2
MAX_SEQUENCES = 300_000
LOG_EVERY = 100
PATIENCE = 5

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

VOCAB_SIZE = None


class PreTokenizedDataset(Dataset):
    def __init__(self, token_ids, seq_len, max_sequences=None):
        total = len(token_ids) // (seq_len + 1)
        if max_sequences:
            total = min(total, max_sequences)
        print(f"  Secuencias: {total:,} (~{total*(seq_len+1):,} tokens)")
        self.n = total
        self.seq_len = seq_len
        self.tokens = token_ids

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        start = i * (self.seq_len + 1)
        seq = self.tokens[start:start + self.seq_len + 1]
        return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)


def make_mask(T, device):
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    tl, tt = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        mask = make_mask(x.size(1), DEVICE)
        logits = model(x, mask)
        loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
        tl += loss.item() * y.numel()
        tt += y.numel()
    model.train()
    return tl / tt


def main():
    global VOCAB_SIZE
    print("=" * 70)
    print("ENTRENAMIENTO BASE — 192 tokens")
    print("=" * 70)

    tok = BigramTokenizer()
    VOCAB_SIZE = tok.vocab_size
    print(f"Tokenizer: {tok}")

    print(f"\n[1/4] Cargando tokens...")
    with open(TOKENIZED_CORPUS, 'rb') as f:
        all_ids = pickle.load(f)
    print(f"  Tokens totales: {len(all_ids):,}")

    print(f"\n[2/4] Dataset...")
    full_ds = PreTokenizedDataset(all_ids, MAX_SEQ_LEN, MAX_SEQUENCES)
    n_val = max(1, int(len(full_ds) * VAL_SPLIT))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))
    print(f"  Train: {n_train:,}  Val: {n_val:,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    print(f"\n[3/4] Modelo...")
    model = BitNetTransformer(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS, num_layers=NUM_LAYERS, ff_dim=FF_DIM,
        max_seq_len=MAX_SEQ_LEN, ple_hidden_dim=PLE_HIDDEN_DIM, dropout=DROPOUT,
    ).to(DEVICE)
    print(f"  Params: {model.count_params():,}")
    print(f"  Batches/época: {len(train_loader):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\n[4/4] Entrenando {EPOCHS} épocas...")
    print("=" * 70)

    best_val = float("inf")
    patience = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        el, et, t0 = 0.0, 0, time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=100)
        for step, (x, y) in enumerate(pbar, 1):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            mask = make_mask(x.size(1), DEVICE)
            logits = model(x, mask)
            loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            el += loss.item() * y.numel()
            et += y.numel()
            if step % LOG_EVERY == 0:
                pbar.set_postfix({"loss": f"{el/et:.4f}"})

        scheduler.step()
        tl = el / et
        vl = evaluate(model, val_loader, criterion)
        dt = time.time() - t0

        print(f"\nÉpoca {epoch}/{EPOCHS}  ({dt:.1f}s)")
        print(f"  Train: {tl:.4f}  Val: {vl:.4f}  PPL: {math.exp(vl):.2f}")

        ckpt = os.path.join(OUTPUT_DIR, f"ep{epoch:02d}_val{vl:.3f}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "val_loss": vl,
            "config": dict(
                vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS,
                num_layers=NUM_LAYERS, ff_dim=FF_DIM,
                max_seq_len=MAX_SEQ_LEN, ple_hidden_dim=PLE_HIDDEN_DIM,
            ),
        }, ckpt)
        print(f"  💾 {ckpt}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if vl < best_val - 1e-4:
            best_val = vl
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"\n🛑 Early stopping")
                break

    print("\n" + "=" * 70)
    print(f"✅ Mejor val loss: {best_val:.4f}  (PPL {math.exp(best_val):.2f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
