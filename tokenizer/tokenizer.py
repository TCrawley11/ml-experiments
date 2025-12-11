import tiktoken

"""
    Wrapper class for the openai tiktoken library.
"""
class Tiktokenizer:
    def __init__(self, enc_name: str = "o200k_base"):
        self.tokenizer = tiktoken.get_encoding(enc_name)


    def encode(self, text, allowed_special: set[str]):
        return self.tokenizer.encode(text, allowed_special=allowed_special)


    def decode(self, tokens, errors="replace"):
        return self.tokenizer.decode(tokens, errors)
    

    def decode_single_token_bytes(self, token):
        return self.tokenizer.decode_single_token_bytes(token)


    def pretty_print_token_to_string(self, tokens):
        print('=' * 30, "token ids to string conversion starting", '=' * 20)
        for token in tokens:
            print(f'token id: {token} -----> {self.tokenizer.decode_single_token_bytes(token)}')
        print('=' * 30, "token ids to string conversion complete", '=' * 20)

    def input_output_pairs(self, tokens, context_size: int = 4):
        step = context_size + 1
        for i in range(0, len(tokens)-step, step-1):
            input = tokens[i:context_size+i]
            target = tokens[context_size+i]
            yield input, target