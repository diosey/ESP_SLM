"""
train.py — BitNet MQA + PLE (7.2M params totales, núcleo ~3M)
"""
import os, time, math, random, gc
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import BitNetTransformer
from tokenizer import BigramTokenizer

torch.set_num_threads(2)

CORPUS_PATH = r"C:\Users\OPTIPLEX 5050\Desktop\IA\Modelo_01\Dataset\corpus_llanero_sin_glosarios.txt"
OUTPUT_DIR  = r"C:\Users\OPTIPLEX 5050\Desktop\IA\Modelo_01\checkpoints_PLE"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Config modelo
EMBED_DIM = 192; NUM_HEADS = 4; NUM_KV_HEADS = 1
NUM_LAYERS = 6; FF_DIM = 384; MAX_SEQ_LEN = 64
PLE_HIDDEN_DIM = 320; DROPOUT = 0.1

# Config entrenamiento
BATCH_SIZE = 32; EPOCHS = 20; LR = 2e-4; WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0; VAL_SPLIT = 0.08; NUM_WORKERS = 0
MAX_STORIES = None; LOG_EVERY = 100; PATIENCE = 5

SEED = 42
random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cpu")
print(f"Device: {DEVICE} | Threads: {torch.get_num_threads()}")
VOCAB_SIZE = None


class StoryDataset(Dataset):
    def __init__(self, text, tok, seq_len, max_stories=None):
        sep = "\n\n" + "=" * 80 + "\n\n"
        stories = [s.strip() for s in text.split(sep) if s.strip()]
        if max_stories: stories = stories[:max_stories]
        print(f"  Historias: {len(stories)}")

        all_ids = []
        for s in stories:
            all_ids.extend(tok.encode(s))
            all_ids.append(tok.EOS_ID)
        print(f"  Tokens: {len(all_ids):,}")

        n = len(all_ids) // (seq_len + 1)
        all_ids = all_ids[:n * (seq_len + 1)]
        t = torch.tensor(all_ids, dtype=torch.long).view(n, seq_len + 1)
        self.inputs = t[:, :-1].contiguous()
        self.targets = t[:, 1:].contiguous()
        print(f"  Secuencias: {n:,}")

    def __len__(self): return len(self.inputs)
    def __getitem__(self, i): return self.inputs[i], self.targets[i]


def make_mask(T):
    return torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_tok = 0.0, 0
    for x, y in loader:
        logits = model(x, make_mask(x.size(1)))
        loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
        total_loss += loss.item() * y.numel()
        total_tok += y.numel()
    model.train()
    return total_loss / total_tok


@torch.no_grad()
def generate(model, tok, prompt, max_new=80, temperature=0.7, top_k=30):
    model.eval()
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    for _ in range(max_new):
        cond = ids[:, -MAX_SEQ_LEN:]
        logits = model(cond, make_mask(cond.size(1)))
        nl = logits[0, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(nl, top_k); nl[nl < v[-1]] = float("-inf")
        nid = torch.multinomial(F.softmax(nl, dim=-1), 1).item()
        ids = torch.cat([ids, torch.tensor([[nid]])], dim=1)
        if nid == tok.EOS_ID: break
    model.train()
    return tok.decode(ids[0].tolist())


def main():
    global VOCAB_SIZE
    print("=" * 70)
    print("ENTRENAMIENTO — BitNet MQA + PLE")
    print("=" * 70)

    tok = BigramTokenizer()
    VOCAB_SIZE = tok.vocab_size
    print(f"\n[1/5] Tokenizer: {tok}")

    print(f"\n[2/5] Corpus...")
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"  {len(text.encode('utf-8'))/1024/1024:.2f} MB")

    print(f"\n[3/5] Dataset...")
    full_ds = StoryDataset(text, tok, MAX_SEQ_LEN, MAX_STORIES)
    n_val = max(1, int(len(full_ds) * VAL_SPLIT))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))
    print(f"  Train: {n_train:,}  Val: {n_val:,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS)

    print(f"\n[4/5] Modelo...")
    model = BitNetTransformer(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS, num_layers=NUM_LAYERS, ff_dim=FF_DIM,
        max_seq_len=MAX_SEQ_LEN, ple_hidden_dim=PLE_HIDDEN_DIM, dropout=DROPOUT,
    )
    print(f"  Params: {model.count_params():,}")
    print(f"  Batches/época: {len(train_loader):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\n[5/5] Entrenando {EPOCHS} épocas...")
    print("=" * 70)

    best_val = float("inf")
    patience = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        el, et, t0 = 0.0, 0, time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=100, mininterval=2.0)
        for step, (x, y) in enumerate(pbar, 1):
            logits = model(x, make_mask(x.size(1)))
            loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            el += loss.item() * y.numel(); et += y.numel()
            if step % LOG_EVERY == 0:
                pbar.set_postfix({"loss": f"{el/et:.4f}"})

        scheduler.step()
        tl = el / et
        vl = evaluate(model, val_loader, criterion)
        dt = time.time() - t0

        print(f"\nÉpoca {epoch}/{EPOCHS}  ({dt:.1f}s)")
        print(f"  Train: {tl:.4f}  Val: {vl:.4f}  PPL: {math.exp(vl):.2f}")
        for p in ["Había una vez", "El llanero"]:
            s = generate(model, tok, p, max_new=60)
            print(f"  📝 '{p}' → {s[:150]}")
        print()

        ckpt = os.path.join(OUTPUT_DIR, f"ep{epoch:02d}_val{vl:.3f}.pt")
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_loss": vl,
                    "config": dict(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                                   num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS,
                                   num_layers=NUM_LAYERS, ff_dim=FF_DIM,
                                   max_seq_len=MAX_SEQ_LEN, ple_hidden_dim=PLE_HIDDEN_DIM)}, ckpt)
        print(f"  💾 {ckpt}")
        gc.collect()

        if vl < best_val - 1e-4:
            best_val = vl; patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"\n🛑 Early stopping"); break

    print("\n" + "=" * 70)
    print(f"✅ Mejor val loss: {best_val:.4f}  (PPL {math.exp(best_val):.2f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
