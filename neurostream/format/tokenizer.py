"""Byte-level and Gemma 4 Unicode BPE tokenizers from GGUF metadata.

No HuggingFace dependency: the vocabulary, merge ranks, special tokens and
chat template are all in the file already. That matters for P6, where the
model is a 146 GB GGUF and downloading a separate tokenizer repo just to turn
text into integers would be silly.
"""
from __future__ import annotations

import functools
import unicodedata
from typing import Iterable

import regex as re

# Pre-tokenizer patterns keyed by the GGUF's `tokenizer.ggml.pre` field,
# transcribed from the matching tokenizer.json. Needs `regex`, not `re`, for
# the Unicode property escapes.
#
# The difference between these is load-bearing: qwen2 splits digits one at a
# time (\p{N}), while the llama3/GPT-4 family groups runs of up to three.
# Using the wrong one silently mis-tokenizes every number in the input.
PRETOKENIZERS = {
    'qwen35': (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
        r"|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    ),
    "gpt-4o": (
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
        r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
        r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    ),
    "qwen2": (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        r"|[^\r\n\p{L}\p{N}]?\p{L}+"
        r"|\p{N}"
        r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
        r"|\s*[\r\n]+"
        r"|\s+(?!\S)"
        r"|\s+"
    ),
    "llama-bpe": (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        r"|[^\r\n\p{L}\p{N}]?\p{L}+"
        r"|\p{N}{1,3}"
        r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
        r"|\s*[\r\n]+"
        r"|\s+(?!\S)"
        r"|\s+"
    ),
}
DEFAULT_PRE = "qwen2"


@functools.lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte <-> printable-codepoint mapping."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class GGUFTokenizer:
    def __init__(self, gguf) -> None:
        md = gguf.metadata
        self.tokens: list[str] = md.get("tokenizer.ggml.tokens", [])
        if not self.tokens:
            raise ValueError("GGUF has no tokenizer.ggml.tokens")
        self.token_types: list[int] = md.get("tokenizer.ggml.token_type", [])
        merges: list[str] = md.get("tokenizer.ggml.merges", [])

        self.vocab: dict[str, int] = {t: i for i, t in enumerate(self.tokens)}
        self.ranks: dict[tuple[str, str], int] = {}
        for i, m in enumerate(merges):
            parts = m.split(" ")
            if len(parts) == 2:
                self.ranks[(parts[0], parts[1])] = i

        self.bos_id = md.get("tokenizer.ggml.bos_token_id")
        self.eos_id = md.get("tokenizer.ggml.eos_token_id")
        self.chat_template = md.get("tokenizer.chat_template")
        self.model_type = md.get('tokenizer.ggml.model', 'gpt2')
        self.is_gemma4 = self.model_type == 'gemma4'
        self.add_bos = self.is_gemma4 and md.get('tokenizer.ggml.add_bos_token', True)
        self.unk_id = md.get('tokenizer.ggml.unknown_token_id', self.vocab.get('<unk>'))
        if self.is_gemma4 and not merges:
            raise ValueError('Gemma 4 GGUF requires tokenizer.ggml.merges')

        self._b2u = bytes_to_unicode()
        self._u2b = {v: k for k, v in self._b2u.items()}
        # The GGUF names its pre-tokenizer variant; honour it.
        self.pre = md.get("tokenizer.ggml.pre", DEFAULT_PRE)
        self._pat = re.compile(
            PRETOKENIZERS.get(self.pre, PRETOKENIZERS[DEFAULT_PRE])
        )
        # Qwen's tokenizer.json declares {"normalizer": {"type": "NFC"}}.
        # Skipping it silently mis-tokenizes any decomposed input.
        self.normalize_nfc = not self.is_gemma4 and self.pre in ('qwen2', 'qwen35')

        # CONTROL tokens (type 3) must be matched literally, never split.
        specials = [
            t
            for t, ty in zip(self.tokens, self.token_types or [])
            if ty == 3 or (self.is_gemma4 and ty == 4)
        ]
        self.special_tokens = set(specials)
        self._special_re = (
            re.compile("(" + "|".join(re.escape(s) for s in sorted(
                specials, key=len, reverse=True)) + ")")
            if specials
            else None
        )

    # -- BPE --------------------------------------------------------------

    @functools.lru_cache(maxsize=65536)
    def _bpe(self, word: str) -> tuple[str, ...]:
        parts = list(word)
        if len(parts) < 2:
            return tuple(parts)
        while True:
            best, best_rank = None, None
            for i in range(len(parts) - 1):
                r = self.ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            parts[best : best + 2] = [parts[best] + parts[best + 1]]
        return tuple(parts)

    def _encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        if self.is_gemma4:
            # Gemma 4 normalizes spaces to the SentencePiece marker but uses
            # Unicode BPE merge ranks, not GPT-2's byte-to-codepoint mapping.
            for tok in self._bpe(text.replace(' ', '\u2581')):
                if tok in self.vocab:
                    out.append(self.vocab[tok])
                else:
                    for byte in tok.encode('utf-8'):
                        idx = self.vocab.get(f'<0x{byte:02X}>', self.unk_id)
                        if idx is None:
                            raise ValueError('Gemma 4 tokenizer has no byte fallback or unknown token')
                        out.append(idx)
            return out
        for piece in self._pat.findall(text):
            mapped = "".join(self._b2u[b] for b in piece.encode("utf-8"))
            for tok in self._bpe(mapped):
                idx = self.vocab.get(tok)
                if idx is None:
                    # Fall back to per-byte tokens rather than dropping input.
                    for ch in tok:
                        bid = self.vocab.get(ch)
                        if bid is not None:
                            out.append(bid)
                else:
                    out.append(idx)
        return out

    def encode(self, text: str, allow_special: bool = True,
               add_special_tokens: bool = True) -> list[int]:
        if self.normalize_nfc:
            text = unicodedata.normalize("NFC", text)
        out: list[int] = []
        chunks = self._special_re.split(text) if allow_special and self._special_re else [text]
        for chunk in chunks:
            if not chunk:
                continue
            if allow_special and chunk in self.special_tokens:
                out.append(self.vocab[chunk])
            else:
                out.extend(self._encode_ordinary(chunk))
        if add_special_tokens and self.add_bos and self.bos_id is not None:
            if not out or out[0] != self.bos_id:
                out.insert(0, self.bos_id)
        return out

    def decode(self, ids: Iterable[int], skip_special: bool = False) -> str:
        return self.decode_bytes(ids, skip_special).decode('utf-8', errors='replace')

    def decode_bytes(self, ids: Iterable[int], skip_special: bool = False) -> bytes:
        """Raw decoded bytes, allowing generation to join split UTF-8 tokens."""
        buf = bytearray()
        for i in ids:
            if i < 0 or i >= len(self.tokens):
                continue
            tok = self.tokens[i]
            if skip_special and tok in self.special_tokens:
                continue
            if tok in self.special_tokens:
                buf.extend(tok.encode("utf-8"))
                continue
            if self.is_gemma4:
                if re.fullmatch(r'<0x[0-9A-Fa-f]{2}>', tok):
                    buf.append(int(tok[3:5], 16))
                else:
                    buf.extend(tok.replace('\u2581', ' ').encode('utf-8'))
                continue
            buf.extend(self._u2b.get(ch, ord(ch) & 0xFF) for ch in tok)
        return bytes(buf)

    @property
    def stop_ids(self) -> set[int]:
        ids = {self.eos_id} if self.eos_id is not None else set()
        markers = ('<turn|>',) if self.is_gemma4 else ('<|im_end|>', '<|endoftext|>')
        return ids | {self.vocab[t] for t in markers if t in self.vocab}

    # -- chat -------------------------------------------------------------

    def apply_chat_template(
        self, messages: list[dict], add_generation_prompt: bool = True, **kwargs
    ) -> str:
        """Render a chat. Uses the model's own Jinja template when jinja2 is
        available, otherwise the ChatML form Qwen3 uses anyway."""
        if self.chat_template:
            try:
                from datetime import datetime
                from jinja2 import Environment
                def raise_exception(message):
                    raise ValueError(message)
                env = Environment(trim_blocks=True, lstrip_blocks=True)
                env.globals.update(strftime_now=lambda fmt: datetime.now().strftime(fmt),
                                   raise_exception=raise_exception)
                context = dict(
                    messages=messages,
                    add_generation_prompt=add_generation_prompt,
                    bos_token=self.tokens[self.bos_id] if self.bos_id is not None else '',
                    eos_token=self.tokens[self.eos_id] if self.eos_id is not None else '',
                )
                context.update(kwargs)
                return env.from_string(self.chat_template).render(**context)
            except ImportError:
                if self.pre == 'qwen35':
                    raise RuntimeError('Qwen3.5 chat formatting requires jinja2; install neurostream[chat]')
                if self.pre == 'gpt-4o' or self.is_gemma4:
                    if self.is_gemma4:
                        raise RuntimeError('Gemma 4 chat formatting requires jinja2; install neurostream[chat]')
                    raise RuntimeError('GPT-OSS chat formatting requires jinja2')
        if self.is_gemma4:
            parts = [self.tokens[self.bos_id]] if self.bos_id is not None else []
            messages = list(messages)
            thinking = kwargs.get('enable_thinking', False)
            system = ''
            if messages and messages[0]['role'] in ('system', 'developer'):
                system = messages.pop(0)['content'].strip()
            if thinking or system:
                parts.append('<|turn>system\n' + ('<|think|>\n' if thinking else '') + system + '<turn|>\n')
            for message in messages:
                role = message['role']
                if role not in ('user', 'assistant') or not isinstance(message['content'], str):
                    raise ValueError('Gemma 4 fallback chat supports text user/assistant messages and an initial system message')
                role = 'model' if role == 'assistant' else role
                parts.append(f"<|turn>{role}\n{message['content'].strip()}<turn|>\n")
            if add_generation_prompt:
                parts.append('<|turn>model\n')
            return ''.join(parts)
        parts = [
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            for m in messages
        ]
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
            if self.pre == 'qwen35':
                parts.append('<think>\n\n</think>\n\n' if kwargs.get('enable_thinking') is False else '<think>\n')
        return "".join(parts)

    def __repr__(self) -> str:
        return (
            f"<GGUFTokenizer pre={self.pre} vocab={len(self.tokens)} "
            f"merges={len(self.ranks)} special={len(self.special_tokens)}>"
        )
