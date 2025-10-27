"""
    Simple bpe tokenizer implementation. Default vocab size of 10,000
"""

from functools import lru_cache

class BPETokenizer:
    def __init__(self):
        # 0 to 255 = single character, 256-10,000 = learned pairs/multi char
        self.vocab = {}
        self.inv_vocab = {}
        # combined bpe pair dicitonary, e.g: (<token_id 69>,<token_id 690>): <merged_token_id 700>)
        self.merged_bpe_pairs = {}
        # use a rank dict for priorities of tokens
        self.rank_dict = {}

    def train(self, text, vocab_size=10000, allowed_special_token='<|endoftext|>'):
        # process the text parameter
        processed_text = []
        for i, ch in enumerate(text):
            if ch == " " and i!=0:
                processed_text.append("Ġ")
            if ch != " ":
                processed_text.append(ch)
        processed_text = "".join(processed_text)

        
        # create the first 0 - 255 vocab
        unique_chars = [chr(i) for i in range(256)]
        unique_chars.extend(
            ch for ch in set(processed_text) if ch not in unique_chars
        )
        if "Ġ" not in unique_chars:
            unique_chars.append("Ġ")
        
        self.vocab = {i: char for i, char in enumarate(unique_chars)}
        self.inv_vocab = {char: i for char, i in self.vocab.items()}

    def get_freqs(self, text):
        """
            Function to count frequencies of input text
        """
        for t in text:
            self.vocab[t] = self.vocab.get(t, 0) + 1


