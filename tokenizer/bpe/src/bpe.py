"""
    Simple bpe tokenizer implementation. Handles regex text splitting.
"""

from functools import lru_cache
import regex as re
from tqdm import tqdm

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

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
        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        # count the pairs and add them to vocab
        merges = {} # {int, int} -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in tqdm(range(num_merges)):
            # count consecutive pairs
            freqs = {}
            for chunk_id in ids:                
                BPETokenizer.update_freqs(chunk_id, freqs) # freqs is updated in-place

            # get the most frequent pair
            if not freqs:
                print("No more pairs to merge. Stopping training.")
                break
            pair = max(freqs, key=freqs.get)
            idx = 256 + i

            ids = [self.merge(chunk_ids, pair, idx) for chunk_ids in ids]
            ids = [self.merge(chunk_ids, pair, idx) for chunk_ids in ids]

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


    def encode(self, text):
        # strings -> integers
        # meant to be used for inference, given text, find the token ids
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        while len(ids) >= 2:
            freqs = self.update_freqs(text, freqs={})
            pair = min(freqs, key=lambda x: self.merges.get(x, float('inf')))
            
            if pair not in self.merges:
                # already have the number of pairs required, no more merging required
                break

            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
            

    def decode(self, ids):
        decoded = []
        for idx in ids:
            if idx in self.vocab:
                decoded.append(self.vocab[idx])
            # add special token handling here
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(decoded)
        text = text_bytes.decode("utf-8", errors="replace")
        return text


    def decode_inference(self, ids):
        # integers -> strings
        # meant to be used at inference time, doesn't process values, just the token id keys
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode(encoding='utf-8', errors='replace')
        return text

    #def encode(self, decoded):
    #    encoded = {}
    #    for k, v in decoded.items():
    #        k1, k2 = k
    #        encoded_k1 = self.vocab.get(k1)
    #        encoded_k2 = self.vocab.get(k2)
    #        encoded_chunk = self.vocab.get(v)
    #        encoded[(encoded_k1, encoded_k2)] = encoded_chunk
    #    return encoded


    @staticmethod
    def update_freqs(text, freqs):
        """
            Function to count frequencies of input text, update freqs in-place
        """
        for pair in zip(text, text[1:]):
            freqs[pair] = freqs.get(pair, 0) + 1
        return freqs


    @staticmethod
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