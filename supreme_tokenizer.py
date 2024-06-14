import glob
import youtokentome as yttm

class SupremeTokenizer:
    def __init__(self, vocab_size):
        # dict of language abbreviation to tokenizer
        self.langs = []
        self.tokenizers = {}
        self.vocab_size = vocab_size

        all_tokenizer_model_files = glob.glob("tokenizers/*.model")

        for model_file in all_tokenizer_model_files:
            lang = model_file.split("/")[-1].split(".")[0].split("_")[-1]
            self.tokenizers[lang] = yttm.BPE(model=model_file)
            self.langs.append(lang)

        self.langs = sorted(self.langs)

    def encode(self, seq, lang, **kwargs):
        local_tokenization = self.tokenizers[lang].encode(seq, **kwargs)

        global_tokenization = []
        for token in local_tokenization:
            if token in [0, 1, 2, 3]:
                global_tokenization.append(token)
            else:
                global_tokenization.append(token + (self.vocab_size * self.langs.index(lang)))

        return global_tokenization
    
    def encode_all(self, seqs, langs, **kwargs):
        return [self.encode(seq, lang, **kwargs) for seq, lang in zip(seqs, langs)]
    
    def decode(self, seq, lang, **kwargs):
        local_seq = []
        for token in seq:
            if token in [0, 1, 2, 3]:
                local_seq.append(token)
            else:
                local_seq.append(token - (self.vocab_size * self.langs.index(lang)))

        return self.tokenizers[lang].decode(local_seq, **kwargs)
    
    def decode_all(self, seqs, langs, **kwargs):
        return [self.decode(seq, lang, **kwargs) for seq, lang in zip(seqs, langs)]
    
    def total_vocab_size(self):
        # 4 is for the special tokens <pad>, <unk>, <bos>, <eos>
        return (self.vocab_size * len(self.tokenizers)) + 4
