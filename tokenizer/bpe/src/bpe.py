"""
    Simple bpe tokenizer implementation. Default vocab size of 5,000
"""

from functools import lru_cache
import regex as re

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def merge(chunk_ids, pair, idx):
    """
        Helper function to replace consecutive occurences of pair with new token idx
    """
    new_ids = []
    i = 0

    while i < len(chunk_ids):
        if chunk_ids[i] == pair[0] and i < len(chunk_ids) - 1 and chunk_ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else: 
            new_ids.append(chunk_ids[i])
            i += 1
    
    return new_ids

class BPETokenizer:
    def __init__(self, pattern):
        # combined bpe pair dicitonary, e.g: (<token_id 69>,<token_id 690>): <merged_token_id 700>)
        self.merges = {}
        self.special_tokes = {}
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern 
        self.compiled_pattern = re.compile(self.pattern)
        self.vocab = self.build_vocab()


    def train(self, text, vocab_size=5000, verbose=False): 
        assert vocab_size >= 2**8
        num_merges = vocab_size - 2**8
        
        # break text into text chunks using regex pattern (pre-process)
        text_chunks = re.findall(self.compiled_pattern, text)

        # token initialization 
        ids = [list(ch.encode("utf-8") for ch in text_chunks)]

        # count the pairs and add them to vocab
        merges = {} # {int, int} -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            # count consecutive pairs
            freqs = {}
            for chunk_id in ids:
                self.update_freqs(chunk_id, freqs)

            # get the most frequent pair
            pair = max(freqs, key=freqs.get)
            idx = 256 + i

            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]

            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"{idx}/{num_merges} pairs merged. {pair} -> idx. {vocab[idx]} occured {freqs.get(pair)} times.")

        self.merges = merges
        self.vocab = vocab


    def build_vocab(self):
        # Build the vocab, start with 256 tokens, then process the pairs
        vocab = {idx: bytes([idx]) for idx in range(256)}

        # make the vocab contain the learned merges
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokes.items():
            vocab[idx] = special.encode("utf-8")

        return vocab


    def update_freqs(self, text, freqs):
        """
            Function to count frequencies of input text, update freqs in-place
        """
        freqs = {} if freqs is None else freqs
        for pair in zip(text[:-1], text[1:]):
            freqs[pair] = freqs.get(pair, 0) + 1
        return freqs

    