import os
import re
import pickle
from collections import Counter
import torch
import pandas as pd
from model import GPTConfig, GPT

# --- Configuration ---
# No longer need a single out_dir, the script will find them automatically.
device = 'cpu' # 'cuda' if you have a GPU
num_samples = 5000 # Number of characters to generate per model
start = " " # Start token
dictionary_path = 'words.txt'
# --- End Configuration ---


def get_top_k_trigrams(text, k=100):
    """Calculates frequencies of all trigrams and returns the top k as a set."""
    trigrams = [text[i:i+3] for i in range(len(text) - 2)]
    trigram_counts = Counter(trigrams)
    return set(trigram for trigram, _ in trigram_counts.most_common(k))


def calculate_word_validity(text, dictionary_set):
    """Calculates the percentage of words in the text that are in the dictionary."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if not words:
        return 0.0
    valid_word_count = sum(1 for w in words if w in dictionary_set)
    return (valid_word_count / len(words)) * 100.0


def evaluate_model(out_dir, dictionary_set, top_train_trigrams):
    """
    Loads a single model, generates text, and calculates evaluation metrics.
    Returns a dictionary with the results, or None if the model can't be loaded.
    """
    print(f"\n=== Evaluating model: {out_dir} ===")
    
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    meta_path = os.path.join(out_dir, 'meta.pkl')

    if not os.path.exists(ckpt_path) or not os.path.exists(meta_path):
        print(f"⚠️ Skipping {out_dir}: Missing checkpoint or meta.pkl file.")
        return None

    # Load model and tokenizer
    checkpoint = torch.load(ckpt_path, map_location=device)
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    vocab_size = meta['vocab_size']
    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda l: ''.join([itos.get(i, '') for i in l])

    # Recreate the model from the config
    config_args = checkpoint['config']
    model_args = {
        k: v for k, v in config_args.items()
        if k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'dropout']
    }
    model_args['vocab_size'] = vocab_size
    config = GPTConfig(**model_args)
    model = GPT(config)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    # Generate sample text
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    with torch.no_grad():
        generated_tokens = model.generate(x, max_new_tokens=num_samples, temperature=0.8, top_k=200)
        generated_text = decode(generated_tokens[0].tolist())

    # Calculate metrics
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
    # 1. Load dictionary and training text trigrams (done only once)
    try:
        with open(dictionary_path, 'r') as f:
            dictionary_set = set(word.strip().lower() for word in f)
        print(f"Loaded {len(dictionary_set)} dictionary words.")
    except FileNotFoundError:
        print(f"Error: Dictionary file not found at {dictionary_path}")
        exit(1)

    with open('data/mark_twain_char/input.txt', 'r', encoding='utf-8') as f:
        train_text = f.read()
    top_train_trigrams = get_top_k_trigrams(train_text, k=100)
    print("Calculated top 100 trigrams from the full Mark Twain training data.")

    # 2. Find all Twain model directories
    model_dirs = sorted([d for d in os.listdir('.') if d.startswith('out-twain-') and os.path.isdir(d)])
    
    if not model_dirs:
        print("⚠️ No model directories found starting with 'out-twain-'. Exiting.")
        exit()
    
    print(f"\nFound model directories to evaluate: {model_dirs}")
    
    # 3. Loop through each model and evaluate it
    results = []
    for out_dir in model_dirs:
        res = evaluate_model(out_dir, dictionary_set, top_train_trigrams)
        if res:
            results.append(res)
            
    # 4. Save results to CSV and print a summary table
    if results:
        df = pd.DataFrame(results)
        df.to_csv('twain_evaluation_results.csv', index=False)
        print("\n✅ Evaluation complete. Results saved to 'twain_evaluation_results.csv'.")
        print("--- Summary ---")
        print(df)
    else:
        print("\n⚠️ No valid models were evaluated.")
