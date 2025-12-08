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
        ids = [id for chunk in text_chunks for id in chunk.encode("utf-8")]

        merges = {} # {int, int} -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in tqdm(range(num_merges)):
            # count consecutive pairs
            freqs = BPETokenizer.update_freqs(ids, {})

            # get the most frequent pair
            if not freqs:
                # this can happen if the text is too small or no pairs can be merged
                print("No more pairs to merge. Stopping training.")
                break
            pair = max(freqs, key=freqs.get)
            idx = 256 + i

            ids = self.merge(ids, pair, idx)

            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"{idx}/{num_merges} pairs merged. {pair} -> {vocab[idx]} occured {freqs.get(pair)} times.")

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


    # not really working, I think this was meant for training?
    def encode(self, text):
        # strings -> integers
        # meant to be used for inference, given text, find the token ids
        text_chunks = re.findall(self.compiled_pattern, text)
        all_ids = []
        for chunk in text_chunks:
            chunk_ids = list(chunk.encode("utf-8"))
            while len(chunk_ids) >= 2:
                # find the next best pair to merge
                stats = BPETokenizer.update_freqs(chunk_ids, {})
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break # nothing else can be merged
                idx = self.merges[pair]
                chunk_ids = self.merge(chunk_ids, pair, idx)
            all_ids.extend(chunk_ids)
        return all_ids


    def encode_single(self, text):
        """
        Take in a single token and return the token id
        Used in inference
        """
        reverse_vocab = {value: key for key, value in self.vocab.items()}
        if isinstance(text, bytes) != True:
            text = text.encode('utf-8')
        return reverse_vocab.get(text)
 

    # not really working, same as the encode method
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

    def decode_single(self, id):
        """
        Take in a single token id and return the token
        Used in inference
        """
        if id is None or id not in self.vocab:
            raise ValueError(f"invalid token id: {id}")
        decoded = self.vocab[id]
        return decoded
        text_bytes = b"".join(decoded)
        if isinstance(text_bytes, bytes):
            return text_bytes.decode("utf-8", errors="replace")
        return text_bytes

    def generate_io_pairs(self, ids, context_size=4):
        # If ids is a list of lists, flatten it.
        if ids and isinstance(ids[0], list):
            ids = [item for sublist in ids for item in sublist]

        # Generate non-overlapping input/output pairs
        # The step size is context_size + 1 to move to the next chunk
        step = context_size + 1
        for i in range(0, len(ids) - step + 1, step):
            ctx = ids[i:i+context_size]
            target = [ids[i+context_size]] # Target is a list with a single token ID
            yield ctx, target


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