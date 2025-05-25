import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ShakespeareDiffusionDataset(Dataset):
    """
    Produces (X, X_s, X_t, t, s) where
      • X      – clean tokens, shape (L,)
      • X_s    – tokens masked with prob. s, shape (L,)
      • X_t    – tokens masked with prob. t, shape (L,)
      • t, s   – scalars in [0,1], |t-s| ≥ 1/32, each returned as shape (1,)
    """
    def __init__(self, bin_path: str, tokenizer, seq_len: int = 1024):
        self.data   = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.N      = len(self.data) - seq_len            # max starting index
        self.L      = seq_len
        self.tok    = tokenizer

        # make sure we have a [MASK] token and record its id
        if self.tok.mask_token is None:
            self.tok.add_special_tokens({"mask_token": "[MASK]"})
        self.mask_id = self.tok.mask_token_id

    def __len__(self):
        # We sample a fresh random window every call, so length can be arbitrary
        return 10**9

    def _corrupt(self, tokens: torch.Tensor, p: float) -> torch.Tensor:
        """Mask each position with probability p (Bernoulli)."""
        mask = torch.bernoulli(torch.full(tokens.shape, p, device=tokens.device)).bool()
        out  = tokens.clone()
        out[mask] = self.mask_id
        return out

    def __getitem__(self, _):
        # 1) random slice of length L
        start = np.random.randint(0, self.N)
        clean = torch.from_numpy(
            self.data[start : start + self.L].astype(np.int64)
        )  # (L,)

        # 2) sample two times with "at least 1/32 spread apart"
        while True:
            t, s = np.random.rand(2)          # U(0,1) each
            if abs(t - s) >= 1 / 32:
                break
        # optional: enforce s > t if you always want the cleaner view second
        if s < t:
            s, t = t, s

        # 3) corrupt
        X_s = self._corrupt(clean, s)
        X_t = self._corrupt(clean, t)

        return (
            clean,            # X      – torch.long
            X_s,              # mask1 – torch.long
            X_t,              # mask2 – torch.long
            torch.tensor([t], dtype=torch.float32),
            torch.tensor([s], dtype=torch.float32),
        )

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

train_ds = ShakespeareDiffusionDataset("train.bin", tokenizer)
val_ds   = ShakespeareDiffusionDataset("val.bin",   tokenizer)

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=32,
    num_workers=2,
    pin_memory=True,
)
