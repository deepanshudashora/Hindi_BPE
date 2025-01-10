import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Iterator, Set
import json
import os
import pandas as pd


class HindiBPETokenizer:
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        # Modified pattern to properly capture Hindi text, spaces, and basic punctuation
        self.pattern = re.compile(r'[\u0900-\u097F]+|[^\u0900-\u097F\s]+|\s+')
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            ' ': 4,  # Explicit space token
        }
        
    def clean_text(self, text: str) -> str:
        """Clean text while preserving Hindi characters, spaces, and basic punctuation."""
        # Keep Hindi characters, spaces, and basic punctuation
        text = re.sub(r'[^\u0900-\u097F\s.,!?-]', '', text)
        # Normalize spaces but preserve them
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def get_initial_vocab(self, texts: Iterator[str]) -> Set[str]:
        """Build initial vocabulary from individual characters and special tokens."""
        chars = set()
        for text in texts:
            cleaned_text = self.clean_text(text)
            for match in self.pattern.finditer(cleaned_text):
                token = match.group()
                if token.strip():
                    chars.update(list(token))
        return chars

    def text_to_word_pieces(self, text: str) -> List[List[str]]:
        """Split text into words and then into character pieces while preserving spaces."""
        cleaned_text = self.clean_text(text)
        word_pieces = []
        
        for match in self.pattern.finditer(cleaned_text):
            token = match.group()
            if token.isspace():
                word_pieces.append([' '])  # Preserve space as a special token
            elif token.strip():
                word_pieces.append(list(token))
        
        return word_pieces

    def get_pair_statistics(self, word_pieces: List[List[str]]) -> Counter:
        """Calculate statistics for adjacent pairs within each word."""
        pairs = Counter()
        for word in word_pieces:
            if len(word) < 2:
                continue
            for i in range(len(word) - 1):
                pairs[tuple(word[i:i+2])] += 1
        return pairs

    def merge_word_pieces(self, word_pieces: List[List[str]], pair: Tuple[str, str]) -> List[List[str]]:
        """Merge all occurrences of a pair in the word pieces."""
        new_word_pieces = []
        for word in word_pieces:
            if len(word) < 2:
                new_word_pieces.append(word)
                continue
                
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and tuple(word[i:i+2]) == pair:
                    new_word.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_pieces.append(new_word)
        return new_word_pieces

    def train(self, texts: Iterator[str]):
        """Train the tokenizer using BPE algorithm."""
        # Initialize vocabulary with special tokens and characters
        chars = self.get_initial_vocab(texts)
        self.vocab = self.special_tokens.copy()
        next_id = len(self.vocab)
        
        # Add individual characters to vocab
        for char in sorted(chars):
            self.vocab[char] = next_id
            next_id += 1

        # Convert texts to word pieces for training
        word_pieces = []
        for text in texts:
            word_pieces.extend(self.text_to_word_pieces(text))

        # Perform BPE merges
        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            pairs = self.get_pair_statistics(word_pieces)
            if not pairs:
                break

            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            new_token = ''.join(best_pair)
            
            if new_token not in self.vocab:
                self.vocab[new_token] = next_id
                next_id += 1
                
            # Record the merge operation
            self.merges[best_pair] = new_token
            
            # Apply the merge throughout the dataset
            word_pieces = self.merge_word_pieces(word_pieces, best_pair)

    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs while properly handling spaces."""
        if not text.strip():
            return []

        tokens = []
        word_pieces = self.text_to_word_pieces(text)
        
        for pieces in word_pieces:
            current_pieces = pieces.copy()
            
            # Apply merges iteratively
            while len(current_pieces) > 1:
                pair = tuple(current_pieces[0:2])
                if pair in self.merges:
                    current_pieces = [self.merges[pair]] + current_pieces[2:]
                else:
                    break
            
            # Convert to token IDs
            for piece in current_pieces:
                if piece in self.vocab:
                    tokens.append(self.vocab[piece])
                else:
                    tokens.append(self.vocab['<UNK>'])
                    
        return tokens

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back into text while properly handling spaces."""
        inv_vocab = {v: k for k, v in self.vocab.items()}
        decoded = []
        
        for id in ids:
            token = inv_vocab.get(id, '<UNK>')
            decoded.append(token)
            
        return ''.join(decoded)

    def save_tokenizer(self, save_dir: str):
        """Save tokenizer configuration to a directory."""
        os.makedirs(save_dir, exist_ok=True)
        config = {
            'vocab_size': self.vocab_size,
            'vocab': self.vocab,
            'merges': {f"{k[0]}|{k[1]}": v for k, v in self.merges.items()},
            'special_tokens': self.special_tokens
        }
        with open(os.path.join(save_dir, 'tokenizer_config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_tokenizer(cls, save_dir: str):
        """Load tokenizer configuration from a directory."""
        with open(os.path.join(save_dir, 'tokenizer_config.json'), 'r', encoding='utf-8') as f:
            config = json.load(f)

        tokenizer = cls(vocab_size=config['vocab_size'])
        tokenizer.vocab = config['vocab']
        tokenizer.special_tokens = config['special_tokens']
        tokenizer.merges = {tuple(k.split('|')): v for k, v in config['merges'].items()}
        return tokenizer


def prepare_dataset(dataset_path: str) -> Iterator[str]:
    """Load dataset in chunks and yield text data."""
    chunk_size = 1000

    # Read training data
    for chunk in pd.read_csv(f"{dataset_path}/train.csv", lineterminator='\n', chunksize=chunk_size):
        for _, row in chunk.iterrows():
            yield f"{row['headline']}\n{row['summary']}\n{row['article']}"

    # Read testing data
    for chunk in pd.read_csv(f"{dataset_path}/test.csv", lineterminator='\n', chunksize=chunk_size):
        for _, row in chunk.iterrows():
            yield f"{row['headline']}\n{row['summary']}\n{row['article']}"


# Example main function
def main():
    dataset_path = "/kaggle/input/hindi-text-short-and-large-summarization-corpus/"
    dataset_iterator = prepare_dataset(dataset_path)

    
    tokenizer = HindiBPETokenizer(vocab_size=5000)
    tokenizer.train(dataset_iterator)
    
    # Test encoding and decoding
    test_text = "नमस्ते दुनिया"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    # Save and load test
    tokenizer.save_tokenizer("hindi_tokenizer_v9")
    loaded_tokenizer = HindiBPETokenizer.load_tokenizer("hindi_tokenizer_v9")
    
    # Verify loaded tokenizer
    encoded_loaded = loaded_tokenizer.encode(test_text)
    decoded_loaded = loaded_tokenizer.decode(encoded_loaded)
    print(f"Decoded (loaded): {decoded_loaded}")


if __name__ == "__main__":
    main()