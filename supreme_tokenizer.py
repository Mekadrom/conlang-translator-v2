import glob
import youtokentome as yttm
import utils

class SupremeTokenizer:
    def __init__(self):
        # dict of language abbreviation to tokenizer
        self.langs = []
        self.tokenizers = {}

        all_tokenizer_model_files = glob.glob("tokenizers/*.model")

        for model_file in all_tokenizer_model_files:
            lang = model_file.split("/")[-1].split(".")[0].split("_")[-1]
            self.tokenizers[lang] = yttm.BPE(model=model_file)
            self.langs.append(lang)

        self.langs = sorted(self.langs)
        self.num_langs = len(self.langs)
        self.num_special_tokens = 4 + self.num_langs

    def encode(self, seq, **kwargs):
        lang_prefix = ''

        for c in seq:
            if c == '>':
                break
            lang_prefix += c

        seq = seq[len(lang_prefix):]

        local_tokenization = self.tokenizers[lang_prefix].encode(seq, **kwargs)

        tokenizer_offset = utils.get_language_tokenizer_offset(lang_prefix)

        global_tokenization = [utils.get_language_indicator_index(lang_prefix)]
        for token in local_tokenization:
            global_tokenization.append(token + tokenizer_offset)

        return global_tokenization
    
    def encode_all(self, seqs, **kwargs):
        return [self.encode(seq, **kwargs) for seq in seqs]
    
    def decode(self, seq, **kwargs):
        lang_prefix_id = seq[0]

        seq = seq[1:]

        lang_prefix = utils.get_language_indicator_from_index(lang_prefix_id)
        tokenizer_offset = utils.get_language_tokenizer_offset(lang_prefix)

        local_seq = []
        for token in seq:
            local_seq.append(token - tokenizer_offset)

        return [f"<{lang_prefix}>"] + self.tokenizers[lang_prefix].decode(local_seq, **kwargs)
    
    def decode_all(self, seqs, **kwargs):
        return [self.decode(seq, **kwargs) for seq in seqs]
    