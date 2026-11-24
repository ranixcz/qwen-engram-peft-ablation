"""
dataset.py — Phase 1 plain causal-LM dataset.
==============================================
Returns {input_ids, labels} only. No masks, no tags, no paired samples.

Sources:
  - HuggingFaceH4/ultrachat_200k  (conversational)
  - Open-Orca/SlimOrca             (reasoning)
  - wikitext/wikitext-103-raw-v1   (prose)
  - gsm8k/main                     (math)
  - cais/mmlu                      (knowledge)

Every sample is:
  - Tokenized from natural text
  - Truncated to max_len - 1 tokens
  - EOS appended
  - Right-padded to max_len with pad_token_id
  - labels = input_ids, except pad positions = -100

Sources stream and can fail gracefully — if a source can't be reached,
it's skipped and the rest still load.
"""
import random
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class EngramFocusedDataset(Dataset):
    def __init__(self, tokenizer, max_len=512, num_samples=20000, seed=42):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id
        self.seed = seed
        self.data = []

        if self.pad_id is None:
            raise ValueError(
                "tokenizer.pad_token_id is None. "
                "Set tokenizer.pad_token before constructing the dataset."
            )
        if self.eos_id is None:
            raise ValueError("tokenizer.eos_token_id is None.")

        print("=" * 60)
        print("Building Phase 1 causal-LM dataset")
        print("=" * 60)
        print(f"  Target samples: {num_samples}")
        print(f"  Max length:     {max_len}")

        source_counts = {}

        # ---- General-purpose (smaller allocation) ----
        source_counts["ultrachat"] = self._load_ultrachat(2000)
        print(f"  ultrachat:    {source_counts['ultrachat']}")

        source_counts["slimorca"] = self._load_slimorca(2000)
        print(f"  slimorca:     {source_counts['slimorca']}")

        source_counts["wikitext"] = self._load_wikitext(1000)
        print(f"  wikitext:     {source_counts['wikitext']}")

        source_counts["gsm8k"] = self._load_gsm8k(2000)
        print(f"  gsm8k:        {source_counts['gsm8k']}")

        # ---- Factual-recall (larger allocation — where Engram helps) ----
        source_counts["mmlu"] = self._load_mmlu(8000)
        print(f"  mmlu:         {source_counts['mmlu']}")

        source_counts["trivia_qa"] = self._load_trivia_qa(4000)
        print(f"  trivia_qa:    {source_counts['trivia_qa']}")

        source_counts["openbookqa"] = self._load_openbookqa(2000)
        print(f"  openbookqa:   {source_counts['openbookqa']}")

        rng = random.Random(seed)
        rng.shuffle(self.data)

        total = len(self.data)
        avg_len = (
            sum(int((s["input_ids"] != self.pad_id).sum()) for s in self.data) / total
            if total > 0 else 0
        )

        print(f"  Total samples: {total}")
        print(f"  Avg real tokens (excl. pad): {avg_len:.1f}")
        if total == 0:
            raise RuntimeError(
                "Dataset is empty. Check network access to HuggingFace datasets."
            )

    # =================================================================
    # Source loaders — each catches its own errors and returns count
    # =================================================================
    def _load_ultrachat(self, limit):
        try:
            ds = load_dataset(
                "HuggingFaceH4/ultrachat_200k",
                split="train_sft",
                streaming=True,
            )
        except Exception as e:
            print(f"    [ultrachat] load failed: {e}")
            return 0

        count = 0
        for item in ds:
            if count >= limit:
                break
            msgs = item.get("messages", [])
            if not msgs:
                continue
            text = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in msgs
            )
            sample = self._tokenize(text)
            if sample is not None:
                self.data.append(sample)
                count += 1
        return count

    def _load_slimorca(self, limit):
        try:
            ds = load_dataset(
                "Open-Orca/SlimOrca", split="train", streaming=True
            )
        except Exception as e:
            print(f"    [slimorca] load failed: {e}")
            return 0

        count = 0
        for item in ds:
            if count >= limit:
                break
            convs = item.get("conversations", [])
            if not convs:
                continue
            text = "\n".join(
                f"{c.get('from', '')}: {c.get('value', '')}" for c in convs
            )
            sample = self._tokenize(text)
            if sample is not None:
                self.data.append(sample)
                count += 1
        return count

    def _load_wikitext(self, limit):
        try:
            ds = load_dataset(
                "wikitext", "wikitext-103-raw-v1",
                split="train", streaming=True,
            )
        except Exception as e:
            print(f"    [wikitext] load failed: {e}")
            return 0

        count = 0
        for item in ds:
            if count >= limit:
                break
            text = (item.get("text", "") or "").strip()
            if len(text) < 50:
                # Skip section headers, blank lines, short stubs.
                continue
            sample = self._tokenize(text)
            if sample is not None:
                self.data.append(sample)
                count += 1
        return count

    def _load_gsm8k(self, limit):
        try:
            ds = load_dataset(
                "gsm8k", "main", split="train", streaming=True
            )
        except Exception as e:
            print(f"    [gsm8k] load failed: {e}")
            return 0

        count = 0
        for item in ds:
            if count >= limit:
                break
            q = (item.get("question", "") or "").strip()
            a = (item.get("answer", "") or "").strip()
            if not q or not a:
                continue
            text = f"Question: {q}\nAnswer: {a}"
            sample = self._tokenize(text)
            if sample is not None:
                self.data.append(sample)
                count += 1
        return count

    def _load_mmlu(self, limit):
        try:
            ds = load_dataset("cais/mmlu", "all", split="auxiliary_train",
                              streaming=True)
        except Exception as e:
            print(f"    [mmlu] load failed: {e}")
            return 0

        letters = "ABCD"
        count = 0
        for item in ds:
            if count >= limit:
                break
            q = (item.get("question", "") or "").strip()
            choices = item.get("choices", [])
            ans_idx = item.get("answer", 0)
            if not q or not choices or not (0 <= ans_idx < len(choices)):
                continue
            opts = "\n".join(
                f"{letters[i]}) {c}" for i, c in enumerate(choices)
                if i < len(letters)
            )
            text = f"Question: {q}\n{opts}\nAnswer: {letters[ans_idx]}"
            sample = self._tokenize(text)
            if sample is not None:
                self.data.append(sample)
                count += 1
        return count

    def _load_trivia_qa(self, limit):
        try:
            ds = load_dataset("trivia_qa", "unfiltered.nocontext",
                              split="train", streaming=True)
        except Exception as e:
            print(f"    [trivia_qa] load failed: {e}")
            return 0
        count = 0
        for item in ds:
            if count >= limit:
                break
            q = (item.get("question", "") or "").strip()
            a = item.get("answer", {})
            ans = (a.get("value", "") if isinstance(a, dict) else "") or ""
            if not q or not ans:
                continue
            text = f"Question: {q}\nAnswer: {ans}"
            sample = self._tokenize(text)
            if sample is not None:
                self.data.append(sample)
                count += 1
        return count

    def _load_openbookqa(self, limit):
        try:
            ds = load_dataset("openbookqa", "main", split="train",
                              streaming=True)
        except Exception as e:
            print(f"    [openbookqa] load failed: {e}")
            return 0
        count = 0
        for item in ds:
            if count >= limit:
                break
            q = (item.get("question_stem", "") or "").strip()
            ans = (item.get("answerKey", "") or "").strip()
            if not q or not ans:
                continue
            sample = self._tokenize(f"Question: {q}\nAnswer: {ans}")
            if sample is not None:
                self.data.append(sample)
                count += 1
        return count

    # =================================================================
    # Tokenization
    # =================================================================

    def _tokenize(self, text):
        if not text or len(text.strip()) < 20:
            return None

        ids = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_len - 1,
        )["input_ids"]

        if len(ids) < 8:
            return None

        ids.append(self.eos_id)

        n_real = len(ids)
        if n_real < self.max_len:
            pad_len = self.max_len - n_real
            ids = ids + [self.pad_id] * pad_len

        labels = list(ids)
        for i in range(n_real, self.max_len):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    # =================================================================
    # Dataset protocol
    # =================================================================
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
