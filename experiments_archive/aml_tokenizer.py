import os
import json
import time
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors

class AMLBPETokenizer:
    def __init__(self, vocab_size=8192):
        self.vocab_size = vocab_size
        self.tokenizer = None
        
    def train(self, corpus_path, save_path="aml_bpe_tokenizer.json"):
        print(f"--- Training BPE Tokenizer on '{corpus_path}' (Vocab size = {self.vocab_size:,}) ---")
        t0 = time.time()
        
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        
        special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True
        )
        
        tokenizer.train([corpus_path], trainer)
        tokenizer.save(save_path)
        self.tokenizer = tokenizer
        
        print(f"BPE Tokenizer trained and saved to '{save_path}' in {time.time() - t0:.2f}s!")
        return self
        
    def load(self, load_path="aml_bpe_tokenizer.json"):
        if os.path.exists(load_path):
            self.tokenizer = Tokenizer.from_file(load_path)
            self.vocab_size = self.tokenizer.get_vocab_size()
            return self
        else:
            raise FileNotFoundError(f"Tokenizer file not found at {load_path}")
            
    def encode(self, text):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded or trained yet.")
        return self.tokenizer.encode(text).ids
        
    def decode(self, token_ids):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded or trained yet.")
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids)

if __name__ == '__main__':
    tokenizer = AMLBPETokenizer().load("aml_bpe_tokenizer.json")
    
    sample_text = "Robert Boulter is an English actor who starred in Valkyria Chronicles III with advanced causal attention mechanisms."
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    
    print("--- TOKENIZER TEST RESULTS ---")
    print(f"Input Text: {sample_text}")
    print(f"Token IDs ({len(encoded)} tokens vs {len(sample_text)} chars): {encoded}")
    print(f"Decoded   : {decoded}")
    print(f"Compression Ratio: {len(sample_text) / len(encoded):.2f}x (1 token ~ {len(sample_text)/len(encoded):.2f} characters)")
