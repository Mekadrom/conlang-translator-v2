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

        if len(seq) < 5:
            return [1] # return <UNK> token

        if seq[3] == '>':
            lang_prefix = seq[:4]
            seq = seq[4:]
        elif seq[4] == '>':
            lang_prefix = seq[:5]
            seq = seq[5:]
        else:
            raise ValueError(f"No language prefix found in sequence! (first 20 chars: {seq[:min(20, len(seq))]})")
        
        lang_prefix = lang_prefix[1:-1] # remove < and >

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

        if len(seq) == 0:
            return "<UNK>"

        lang_prefix = utils.get_language_indicator_from_index(lang_prefix_id)
        tokenizer_offset = utils.get_language_tokenizer_offset(lang_prefix)

        local_seq = []
        for token in seq:
            local_seq.append(token - tokenizer_offset)

        try:
            return [f"<{lang_prefix}>"] + self.tokenizers[lang_prefix].decode(local_seq, **kwargs)
        except Exception as e:
            print(e)
            print(f"Error decoding sequence for language {lang_prefix} with tokens {local_seq}")
            return "<UNK>"
    
    def decode_all(self, seqs, **kwargs):
        return [self.decode(seq, **kwargs) for seq in seqs]
    