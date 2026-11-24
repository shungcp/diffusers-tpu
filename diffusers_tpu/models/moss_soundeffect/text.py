"""MOSS-SoundEffect-v2's prompt path: cleaning, tokenizing, Qwen3-1.7B.

The official prompter (DiffSynth's ``WanPrompter``) cleans the text with
``ftfy.fix_text`` + ``html.unescape`` + whitespace collapsing, tokenizes it
right-padded to 512 tokens, runs Qwen3 with the attention mask and keeps the
last hidden state with the padded rows zeroed.

``ftfy`` is not in this repository's environment, and it is not a no-op:
by default it turns full-width punctuation into ASCII (a Chinese prompt's
"，" and "！"), straightens curly quotes and splits Latin ligatures, all of
which change the tokens. ``fix_text`` below uses ftfy when it is installed
and otherwise applies the same steps of its default configuration except
mojibake repair (``fix_encoding``), which only matters for text that was
already decoded with the wrong codec.

Qwen3 is causal and the padding is on the right, so the valid rows do not
depend on the padding: the encoder runs unmasked, the padded rows are
zeroed afterwards exactly as the official code does.
"""
import html
import re
import unicodedata

TEXT_LEN = 512

# ftfy 6.3 defaults, minus fix_encoding / restore_byte_a0 / replace_lossy_sequences /
# decode_inconsistent_utf8 (mojibake repair).
_TERMINAL_ESCAPES = re.compile(r"\033\[((?:\d|;)*)([a-zA-Z])")
_SINGLE_QUOTES = re.compile("[ʼ‘-‛]")
_DOUBLE_QUOTES = re.compile("[“-‟]")
_LINE_BREAKS = re.compile("\r\n|\r| | |\u0085")
_LIGATURES = {
    "Ĳ": "IJ", "ĳ": "ij", "ŉ": "ʼn", "Ǆ": "DŽ", "ǅ": "Dž",
    "ǆ": "dž", "Ǉ": "LJ", "ǈ": "Lj", "ǉ": "lj", "Ǌ": "NJ",
    "ǋ": "Nj", "ǌ": "nj", "Ǳ": "DZ", "ǲ": "Dz", "ǳ": "dz",
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "ſt", "ﬆ": "st",
}
_CONTROL_CHARS = (
    [chr(c) for c in range(0x00, 0x09)] + ["\x0b"] + [chr(c) for c in range(0x0e, 0x20)]
    + ["\x7f", "﻿", "￹", "￺", "￻"]
    + [chr(c) for c in range(0x206a, 0x2070)]
    + [chr(c) for c in range(0x1d173, 0x1d17b)]
    + [chr(c) for c in range(0xe0000, 0xe0080)]
)
_CONTROL_TABLE = {ord(c): None for c in _CONTROL_CHARS}


def _width(ch):
  """ftfy's character-width fix: full-width ASCII and the ideographic space to
  their plain forms, half-width katakana to full width."""
  code = ord(ch)
  if 0xff01 <= code <= 0xff5e or code == 0x3000 or 0xff61 <= code <= 0xffdc or 0xffe8 <= code <= 0xffee:
    return unicodedata.normalize("NFKC", ch)
  return ch


def _fix_c1(text):
  def repl(m):
    try:
      return m.group(0).encode("latin-1").decode("windows-1252")
    except UnicodeDecodeError:
      return m.group(0)
  return re.sub("[\x80-\x9f]", repl, text)


def fix_text(text):
  """``ftfy.fix_text`` when ftfy is installed, else ``fix_text_without_ftfy``."""
  try:
    import ftfy
  except ImportError:
    return fix_text_without_ftfy(text)
  return ftfy.fix_text(text)


def fix_text_without_ftfy(text):
  """ftfy 6.3's default fixes except mojibake repair (see the module docstring)."""
  if "<" not in text:  # unescape_html="auto"
    text = html.unescape(text)
  text = _TERMINAL_ESCAPES.sub("", text)
  text = _fix_c1(text)
  text = "".join(_LIGATURES.get(c, c) for c in text)
  text = "".join(_width(c) for c in text)
  text = _SINGLE_QUOTES.sub("'", text)
  text = _DOUBLE_QUOTES.sub('"', text)
  text = _LINE_BREAKS.sub("\n", text)
  text = text.translate(_CONTROL_TABLE)
  return unicodedata.normalize("NFC", text)


def clean(text):
  """The prompter's ``whitespace`` cleaning."""
  text = html.unescape(html.unescape(fix_text(text))).strip()
  return re.sub(r"\s+", " ", text).strip()


def format_prompt(prompt, seconds, append_duration=True):
  """``MossSoundEffectPipeline``'s prompt: ``"<prompt> duration: <s>s"``."""
  prompt = prompt.strip()
  return f"{prompt} duration: {round(float(seconds), 1):.1f}s" if append_duration else prompt


def tokenize(tokenizer, prompts):
  """(ids [B, 512] int32, lengths) -- right padded to 512, truncated."""
  enc = tokenizer([clean(p) for p in prompts], return_tensors="np", padding="max_length",
                  truncation=True, max_length=TEXT_LEN, add_special_tokens=True)
  return enc["input_ids"].astype("int32"), enc["attention_mask"].sum(-1).astype("int32")
