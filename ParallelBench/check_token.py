#!/usr/bin/env python3
"""
Check what token 126081 is in the LLaDA tokenizer.
"""

from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

# Check specific tokens
tokens_to_check = [126081, 126336, 126464-1]  # Token in question, mask token, last token

print("Token Information:")
print("="*60)

for token_id in tokens_to_check:
    try:
        decoded = tokenizer.decode([token_id])
        print(f"Token {token_id}: '{decoded}'")
        print(f"  Repr: {repr(decoded)}")
    except Exception as e:
        print(f"Token {token_id}: Error - {e}")

# Check if it's a special token
print("\n" + "="*60)
print("Special Tokens:")
print(f"  PAD token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
print(f"  EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
print(f"  BOS token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
print(f"  UNK token: {tokenizer.unk_token} (id={tokenizer.unk_token_id})")
print(f"  MASK token: id={126336}")

print("\n" + "="*60)
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Is 126081 in special_tokens_map? {126081 in tokenizer.all_special_ids}")

