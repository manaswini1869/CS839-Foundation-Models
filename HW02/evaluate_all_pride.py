import os
import re
import pickle
from collections import Counter
import torch
import pandas as pd
from model import GPTConfig, GPT

# --- Configuration ---
device = 'cpu'  # 'cuda' if you have GPU
num_samples = 5000
start = " "
dictionary_path = 'words.txt'
data_dir = 'data/pride_char'  # <- update if your pride data folder name differs
# --- End Configuration ---


def get_top_k_trigrams(text, k=100):
    trigrams = [text[i:i+3] for i in range(len(text) - 2)]
    trigram_counts = Counter(trigrams)
    return set(trigram for trigram, _ in trigram_counts.most_common(k))


def calculate_word_validity(text, dictionary_set):
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if not words:
        return 0.0
    valid_word_count = sum(1 for w in words if w in dictionary_set)
    return (valid_word_count / len(words)) * 100.0


def evaluate_model(out_dir, train_text, top_train_trigrams, dictionary_set):
    print(f"\n=== Evaluating model: {out_dir} ===")

    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        print(f"⚠️ Skipping {out_dir}: no checkpoint found.")
        return None

    checkpoint = torch.load(ckpt_path, map_location=device)

    # Load tokenizer metadata
    # Try to load meta.pkl from the model's corresponding data directory
    possible_meta_paths = [
        os.path.join(out_dir, 'meta.pkl'),
        os.path.join('data', 'pride_char', 'meta.pkl'),
        os.path.join('data', 'shakespeare_char', 'meta.pkl'),
    ]

    meta_path = None
    for path in possible_meta_paths:
        if os.path.exists(path):
            meta_path = path
            break

    if meta_path is None:
        print(f"⚠️ No meta.pkl found for {out_dir}, skipping.")
        return None

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    stoi, itos = meta['stoi'], meta['itos']
    vocab_size = meta['vocab_size']
    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda l: ''.join([itos[i] for i in l if i in itos])

    # Prepare model config
    config_dict = checkpoint['config']
    allowed_keys = {'block_size', 'vocab_size', 'n_layer', 'n_head', 'n_embd', 'dropout', 'bias'}
    filtered_config = {k: v for k, v in config_dict.items() if k in allowed_keys}
    filtered_config['vocab_size'] = vocab_size
    filtered_config['block_size'] = 64

    config = GPTConfig(**filtered_config)
    model = GPT(config)

    # Load weights
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)

    # Generate text
    start_ids = encode(start)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        generated_tokens = model.generate(x, num_samples, temperature=0.8, top_k=200)
        generated_text = decode(generated_tokens[0].tolist())

    # Metrics
    top_gen_trigrams = get_top_k_trigrams(generated_text, k=100)
    overlap = top_train_trigrams.intersection(top_gen_trigrams)
    trigram_overlap_metric = (len(overlap) / 100.0) * 100.0

    validity_rate_metric = calculate_word_validity(generated_text, dictionary_set)

    print(f"Trigram Overlap: {trigram_overlap_metric:.2f}% | Word Validity: {validity_rate_metric:.2f}%")

    return {
        "model": out_dir,
        "trigram_overlap": trigram_overlap_metric,
        "word_validity": validity_rate_metric
    }


if __name__ == "__main__":
    # Load dictionary
    with open(dictionary_path, 'r') as f:
        dictionary_set = set(word.strip().lower() for word in f)
    print(f"Loaded {len(dictionary_set)} dictionary words.")

    # Load training text (Pride and Prejudice)
    train_text_path = os.path.join(data_dir, 'input.txt')
    with open(train_text_path, 'r', encoding='utf-8') as f:
        train_text = f.read()
    top_train_trigrams = get_top_k_trigrams(train_text, k=100)

    # Find all out directories
    model_dirs = sorted([d for d in os.listdir('.') if d.startswith('out-pride-char')])
    print(f"\nFound model directories: {model_dirs}")

    # Evaluate each model
    results = []
    for out_dir in model_dirs:
        if out_dir == 'out-pride-char-0k':
            continue
        res = evaluate_model(out_dir, train_text, top_train_trigrams, dictionary_set)
        if res:
            results.append(res)

    # Save results to CSV and print summary
    if results:
        df = pd.DataFrame(results)
        df.to_csv('evaluation_results.csv', index=False)
        print("\n✅ Evaluation complete. Results saved to 'evaluation_results.csv'.")
        print(df)
    else:
        print("⚠️ No valid models were evaluated.")

