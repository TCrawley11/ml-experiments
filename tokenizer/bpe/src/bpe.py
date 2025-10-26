"""
    Simple bpe tokenizer implementation. Default vocab size of 10,000
"""
class BPETokenizer:
    def __init__(self):
        # 0 to 255 = single character, 256-10,000 = learned pairs/multi char
        self.vocab = {}

        # translate token_ids to their char(s) representations
        self.inv_vocab = {}

        # combined bpe pair dicitonary, e.g: (<token_id 69>,<token_id 690>): <merged_token_id 700>)
        self.merged_bpe_pairs = {}

        # use a rank dict for priorities of tokens
        self.rank_dict = {}

    def train(self, text, vocab_size=10000, allowed_special_token='<|endoftext|>'):
        # process the text parameter
        
        # create the first 0 - 255 vocab
        for i in range(256):
            self.vocab[i] = chr(i)
            self.inv_vocab[chr(i)] = i

        print(f"First 256 characters: {self.vocab}, length check: {len(self.vocab)}") 
