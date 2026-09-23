"""Simple, transparent SMILES tokenizer for the teaching prototype.

Character-level, not chemistry-aware (no atom/bond-token vocabulary) -- this
keeps the tokenizer trivial to read and debug, at the cost of the model
having to learn SMILES syntax itself rather than getting it for free."""
from collections import Counter
import json

SPECIAL = ["[PAD]","[UNK]","[MASK]","[CLS]"]

class SmilesTokenizer:
    """Character vocabulary fit on the training SMILES, plus the special
    tokens every masked-language-model needs: [PAD]/[UNK]/[MASK]/[CLS]."""

    def __init__(self, vocab=None):
        self.vocab = vocab or {t:i for i,t in enumerate(SPECIAL)}

    @property
    def pad_id(self): return self.vocab["[PAD]"]
    @property
    def unk_id(self): return self.vocab["[UNK]"]
    @property
    def mask_id(self): return self.vocab["[MASK]"]
    @property
    def cls_id(self): return self.vocab["[CLS]"]

    def fit(self, smiles, max_vocab=512):
        """Build the vocabulary from the most frequent characters across
        `smiles` (fit on the training split only, never validation)."""
        counts = Counter(ch for smi in smiles for ch in str(smi))
        for ch,_ in counts.most_common(max_vocab-len(self.vocab)):
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)
        return self

    def encode(self, smiles, max_length):
        """[CLS] + one id per character, truncated/right-padded to `max_length`.
        The [CLS] position is what the model reads out as the SMILES embedding."""
        ids = [self.cls_id] + [self.vocab.get(ch,self.unk_id) for ch in str(smiles)]
        ids = ids[:max_length]
        return ids + [self.pad_id]*(max_length-len(ids))

    def save(self, path):
        """Persist the vocabulary as JSON, so a saved checkpoint can rebuild
        an identical tokenizer at inference time (see core/checkpoint.py)."""
        with open(path,"w",encoding="utf-8") as f:
            json.dump(self.vocab,f,indent=2)
