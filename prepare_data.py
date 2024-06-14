from datasets import load_dataset
from tqdm import tqdm

import glob
import os
import youtokentome as yttm

def download_dataset(path, src_lang, tgt_lang, name=None, manual_split=False, collation_fn=None):
    dataset = load_dataset(path, name, cache_dir='data/', trust_remote_code=True)

    def save_to_file(data, src_filename, tgt_filename):
        src_data_path = os.path.join('data', src_filename)
        tgt_data_path = os.path.join('data', tgt_filename)

        if os.path.exists(src_data_path) and os.path.exists(tgt_data_path):
            if os.path.getsize(src_data_path) > 0 and os.path.getsize(tgt_data_path) > 0:
                # skip if data already exists
                print(f"Skipping {src_data_path} and {tgt_data_path}")
                return
            else:
                # delete if data is empty due to previous error
                os.remove(src_data_path)
                os.remove(tgt_data_path)
        
        with open(src_data_path, 'a', encoding='utf-8') as src_data_file, open(tgt_data_path, 'a', encoding='utf-8') as tgt_data_file:
            for example in tqdm(data, unit=' examples', total=len(data)):
                if collation_fn is not None:
                    example = collation_fn(example)

                src_data_file.write(f"<{src_lang}>{example['translation'][src_lang]}\n")
                tgt_data_file.write(f"<{tgt_lang}>{example['translation'][tgt_lang]}\n")


    # datasets have to be paired up
    if manual_split:
        dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
        dataset['validation'] = dataset['test']

    pair_name = f"{src_lang}-{tgt_lang}"
    train_name = f"train_{pair_name}"
    validation_name = f"validation_{pair_name}"

    print(f"Saving to file: {path}/{pair_name}")

    save_to_file(dataset['train'], f"{train_name}.{src_lang}", f"{train_name}.{tgt_lang}")
    save_to_file(dataset['validation'], f"{validation_name}.{src_lang}", f"{validation_name}.{tgt_lang}")

def download_base_traindata():
    # total # of languages represented: 34

    download_dataset("may-ohta/jparacrawl", "en", "ja", 'en-ja', manual_split=True, collation_fn=lambda example: { "translation": { "en": example["translation"]["en"], "ja": example["translation"]["ja"] } })
    # download_dataset("may-ohta/jparacrawl", "zh", "ja", 'zh-ja', manual_split=True, collation_fn=lambda example: { "translation": { "zh": example["translation"]["zh"], "ja": example["translation"]["ja"] } })

    download_dataset("wmt/wmt19", "cs", "en", "cs-en")
    download_dataset("wmt/wmt19", "de", "en", "de-en")
    download_dataset("wmt/wmt19", "fi", "en", "fi-en")
    download_dataset("wmt/wmt19", "fr", "de", "fr-de")
    download_dataset("wmt/wmt19", "gu", "en", "gu-en")
    download_dataset("wmt/wmt19", "kk", "en", "kk-en")
    download_dataset("wmt/wmt19", "lt", "en", "lt-en")
    download_dataset("wmt/wmt19", "ru", "en", "ru-en")
    download_dataset("wmt/wmt19", "zh", "en", "zh-en")

    download_dataset("wmt/wmt18", "et", "en", "et-en")
    download_dataset("wmt/wmt18", "tr", "en", "tr-en")

    download_dataset("wmt/wmt17", "lv", "en", "lv-en")

    download_dataset("wmt/wmt16", "ro", "en", "ro-en")

    download_dataset("wmt/wmt15", "fr", "en", "fr-en")

    download_dataset("wmt/wmt14", "hi", "en", "hi-en")

    def eng_to_en(example):
        return { "translation": { "en" if key == "eng" else key: value for key, value in example['translation'].items() } }
    
    def fra_to_fr(example):
        return { "translation": { "fr" if key == "fra" else key: value for key, value in example['translation'].items() } }

    download_dataset("allenai/wmt22_african", "afr", "en", "afr-eng", manual_split=True, collation_fn=eng_to_en)
    download_dataset("allenai/wmt22_african", "afr", "som", "afr-som", manual_split=True)

    download_dataset("allenai/wmt22_african", "amh", "en", "amh-eng", manual_split=True, collation_fn=eng_to_en)
    download_dataset("allenai/wmt22_african", "amh", "fr", "amh-fra", manual_split=True, collation_fn=fra_to_fr)
    download_dataset("allenai/wmt22_african", "amh", "nya", "amh-nya", manual_split=True)
    download_dataset("allenai/wmt22_african", "amh", "orm", "amh-orm", manual_split=True)
    download_dataset("allenai/wmt22_african", "amh", "sna", "amh-sna", manual_split=True)
    download_dataset("allenai/wmt22_african", "amh", "som", "amh-som", manual_split=True)
    download_dataset("allenai/wmt22_african", "amh", "ssw", "amh-ssw", manual_split=True)
    download_dataset("allenai/wmt22_african", "amh", "swh", "amh-swh", manual_split=True)
    download_dataset("allenai/wmt22_african", "amh", "tsn", "amh-tsn", manual_split=True)
    download_dataset("allenai/wmt22_african", "amh", "tso", "amh-tso", manual_split=True)
    download_dataset("allenai/wmt22_african", "amh", "umb", "amh-umb", manual_split=True)
    download_dataset("allenai/wmt22_african", "amh", "swh", "amh-swh", manual_split=True)
    download_dataset("allenai/wmt22_african", "amh", "xho", "amh-xho", manual_split=True)
    download_dataset("allenai/wmt22_african", "amh", "swh", "amh-swh", manual_split=True)
    download_dataset("allenai/wmt22_african", "amh", "zul", "amh-zul", manual_split=True)

    download_dataset("allenai/wmt22_african", "en", "fuv", "eng-fuv", manual_split=True, collation_fn=eng_to_en)
    download_dataset("allenai/wmt22_african", "en", "hau", "eng-hau", manual_split=True, collation_fn=eng_to_en)
    download_dataset("allenai/wmt22_african", "en", "ibo", "eng-ibo", manual_split=True, collation_fn=eng_to_en)
    download_dataset("allenai/wmt22_african", "en", "kam", "eng-kam", manual_split=True, collation_fn=eng_to_en)
    download_dataset("allenai/wmt22_african", "en", "kin", "eng-kin", manual_split=True, collation_fn=eng_to_en)
    download_dataset("allenai/wmt22_african", "en", "lin", "eng-lin", manual_split=True, collation_fn=eng_to_en)
    download_dataset("allenai/wmt22_african", "en", "lug", "eng-lug", manual_split=True, collation_fn=eng_to_en)
    download_dataset("allenai/wmt22_african", "en", "luo", "eng-luo", manual_split=True, collation_fn=eng_to_en)
    download_dataset("allenai/wmt22_african", "en", "nso", "eng-nso", manual_split=True, collation_fn=eng_to_en)
    download_dataset("allenai/wmt22_african", "en", "nya", "eng-nya", manual_split=True, collation_fn=eng_to_en)
    download_dataset("allenai/wmt22_african", "en", "orm", "eng-orm", manual_split=True, collation_fn=eng_to_en)
    download_dataset("allenai/wmt22_african", "en", "sna", "eng-sna", manual_split=True, collation_fn=eng_to_en)
    download_dataset("allenai/wmt22_african", "en", "som", "eng-som", manual_split=True, collation_fn=eng_to_en)
    download_dataset("allenai/wmt22_african", "en", "ssw", "eng-ssw", manual_split=True, collation_fn=eng_to_en)
    download_dataset("allenai/wmt22_african", "en", "swh", "eng-swh", manual_split=True, collation_fn=eng_to_en)
    download_dataset("allenai/wmt22_african", "en", "tsn", "eng-tsn", manual_split=True, collation_fn=eng_to_en)
    download_dataset("allenai/wmt22_african", "en", "tso", "eng-tso", manual_split=True, collation_fn=eng_to_en)
    download_dataset("allenai/wmt22_african", "en", "umb", "eng-umb", manual_split=True, collation_fn=eng_to_en)
    download_dataset("allenai/wmt22_african", "en", "wol", "eng-wol", manual_split=True, collation_fn=eng_to_en)
    download_dataset("allenai/wmt22_african", "en", "xho", "eng-xho", manual_split=True, collation_fn=eng_to_en)
    download_dataset("allenai/wmt22_african", "en", "swh", "eng-swh", manual_split=True, collation_fn=eng_to_en)
    download_dataset("allenai/wmt22_african", "en", "yor", "eng-yor", manual_split=True, collation_fn=eng_to_en)
    download_dataset("allenai/wmt22_african", "en", "zul", "eng-zul", manual_split=True, collation_fn=eng_to_en)

    download_dataset("allenai/wmt22_african", "fr", "hau", "fra-hau", manual_split=True, collation_fn=fra_to_fr)
    download_dataset("allenai/wmt22_african", "fr", "ibo", "fra-ibo", manual_split=True, collation_fn=fra_to_fr)
    download_dataset("allenai/wmt22_african", "fr", "kam", "fra-kam", manual_split=True, collation_fn=fra_to_fr)
    download_dataset("allenai/wmt22_african", "fr", "kin", "fra-kin", manual_split=True, collation_fn=fra_to_fr)
    download_dataset("allenai/wmt22_african", "fr", "lin", "fra-lin", manual_split=True, collation_fn=fra_to_fr)
    download_dataset("allenai/wmt22_african", "fr", "lug", "fra-lug", manual_split=True, collation_fn=fra_to_fr)
    download_dataset("allenai/wmt22_african", "fr", "luo", "fra-luo", manual_split=True, collation_fn=fra_to_fr)
    download_dataset("allenai/wmt22_african", "fr", "nso", "fra-nso", manual_split=True, collation_fn=fra_to_fr)
    download_dataset("allenai/wmt22_african", "fr", "nya", "fra-nya", manual_split=True, collation_fn=fra_to_fr)
    download_dataset("allenai/wmt22_african", "fr", "orm", "fra-orm", manual_split=True, collation_fn=fra_to_fr)
    download_dataset("allenai/wmt22_african", "fr", "som", "fra-som", manual_split=True, collation_fn=fra_to_fr)
    download_dataset("allenai/wmt22_african", "fr", "ssw", "fra-ssw", manual_split=True, collation_fn=fra_to_fr)
    download_dataset("allenai/wmt22_african", "fr", "swh", "fra-swh", manual_split=True, collation_fn=fra_to_fr)
    download_dataset("allenai/wmt22_african", "fr", "tsn", "fra-tsn", manual_split=True, collation_fn=fra_to_fr)
    download_dataset("allenai/wmt22_african", "fr", "tso", "fra-tso", manual_split=True, collation_fn=fra_to_fr)
    download_dataset("allenai/wmt22_african", "fr", "umb", "fra-umb", manual_split=True, collation_fn=fra_to_fr)
    download_dataset("allenai/wmt22_african", "fr", "wol", "fra-wol", manual_split=True, collation_fn=fra_to_fr)
    download_dataset("allenai/wmt22_african", "fr", "xho", "fra-xho", manual_split=True, collation_fn=fra_to_fr)
    download_dataset("allenai/wmt22_african", "fr", "zul", "fra-zul", manual_split=True, collation_fn=fra_to_fr)

    download_dataset("allenai/wmt22_african", "fuv", "hau", "fuv-hau", manual_split=True)
    download_dataset("allenai/wmt22_african", "fuv", "ibo", "fuv-ibo", manual_split=True)
    download_dataset("allenai/wmt22_african", "fuv", "kam", "fuv-kam", manual_split=True)
    download_dataset("allenai/wmt22_african", "fuv", "kin", "fuv-kin", manual_split=True)
    download_dataset("allenai/wmt22_african", "fuv", "lug", "fuv-lug", manual_split=True)
    download_dataset("allenai/wmt22_african", "fuv", "luo", "fuv-luo", manual_split=True)
    download_dataset("allenai/wmt22_african", "fuv", "nso", "fuv-nso", manual_split=True)
    download_dataset("allenai/wmt22_african", "fuv", "nya", "fuv-nya", manual_split=True)
    download_dataset("allenai/wmt22_african", "fuv", "orm", "fuv-orm", manual_split=True)
    download_dataset("allenai/wmt22_african", "fuv", "sna", "fuv-sna", manual_split=True)
    download_dataset("allenai/wmt22_african", "fuv", "som", "fuv-som", manual_split=True)
    download_dataset("allenai/wmt22_african", "fuv", "ssw", "fuv-ssw", manual_split=True)
    download_dataset("allenai/wmt22_african", "fuv", "swh", "fuv-swh", manual_split=True)
    download_dataset("allenai/wmt22_african", "fuv", "tsn", "fuv-tsn", manual_split=True)
    download_dataset("allenai/wmt22_african", "fuv", "tso", "fuv-tso", manual_split=True)
    download_dataset("allenai/wmt22_african", "fuv", "umb", "fuv-umb", manual_split=True)
    download_dataset("allenai/wmt22_african", "fuv", "xho", "fuv-xho", manual_split=True)
    download_dataset("allenai/wmt22_african", "fuv", "yor", "fuv-yor", manual_split=True)
    download_dataset("allenai/wmt22_african", "fuv", "zul", "fuv-zul", manual_split=True)

    download_dataset("allenai/wmt22_african", "hau", "ibo", "hau-ibo", manual_split=True)
    download_dataset("allenai/wmt22_african", "hau", "kam", "hau-kam", manual_split=True)
    download_dataset("allenai/wmt22_african", "hau", "kin", "hau-kin", manual_split=True)
    download_dataset("allenai/wmt22_african", "hau", "lug", "hau-lug", manual_split=True)
    download_dataset("allenai/wmt22_african", "hau", "luo", "hau-luo", manual_split=True)
    download_dataset("allenai/wmt22_african", "hau", "nso", "hau-nso", manual_split=True)
    download_dataset("allenai/wmt22_african", "hau", "nya", "hau-nya", manual_split=True)
    download_dataset("allenai/wmt22_african", "hau", "orm", "hau-orm", manual_split=True)
    download_dataset("allenai/wmt22_african", "hau", "sna", "hau-sna", manual_split=True)
    download_dataset("allenai/wmt22_african", "hau", "som", "hau-som", manual_split=True)
    download_dataset("allenai/wmt22_african", "hau", "ssw", "hau-ssw", manual_split=True)
    download_dataset("allenai/wmt22_african", "hau", "swh", "hau-swh", manual_split=True)
    download_dataset("allenai/wmt22_african", "hau", "tsn", "hau-tsn", manual_split=True)
    download_dataset("allenai/wmt22_african", "hau", "tso", "hau-tso", manual_split=True)
    download_dataset("allenai/wmt22_african", "hau", "umb", "hau-umb", manual_split=True)
    download_dataset("allenai/wmt22_african", "hau", "xho", "hau-xho", manual_split=True)
    download_dataset("allenai/wmt22_african", "hau", "yor", "hau-yor", manual_split=True)
    download_dataset("allenai/wmt22_african", "hau", "zul", "hau-zul", manual_split=True)

    download_dataset("allenai/wmt22_african", "ibo", "kam", "ibo-kam", manual_split=True)
    download_dataset("allenai/wmt22_african", "ibo", "kin", "ibo-kin", manual_split=True)
    download_dataset("allenai/wmt22_african", "ibo", "lug", "ibo-lug", manual_split=True)
    download_dataset("allenai/wmt22_african", "ibo", "luo", "ibo-luo", manual_split=True)
    download_dataset("allenai/wmt22_african", "ibo", "nso", "ibo-nso", manual_split=True)
    download_dataset("allenai/wmt22_african", "ibo", "nya", "ibo-nya", manual_split=True)
    download_dataset("allenai/wmt22_african", "ibo", "orm", "ibo-orm", manual_split=True)
    download_dataset("allenai/wmt22_african", "ibo", "sna", "ibo-sna", manual_split=True)
    download_dataset("allenai/wmt22_african", "ibo", "som", "ibo-som", manual_split=True)
    download_dataset("allenai/wmt22_african", "ibo", "ssw", "ibo-ssw", manual_split=True)
    download_dataset("allenai/wmt22_african", "ibo", "swh", "ibo-swh", manual_split=True)
    download_dataset("allenai/wmt22_african", "ibo", "tsn", "ibo-tsn", manual_split=True)
    download_dataset("allenai/wmt22_african", "ibo", "tso", "ibo-tso", manual_split=True)
    download_dataset("allenai/wmt22_african", "ibo", "umb", "ibo-umb", manual_split=True)
    download_dataset("allenai/wmt22_african", "ibo", "xho", "ibo-xho", manual_split=True)
    download_dataset("allenai/wmt22_african", "ibo", "yor", "ibo-yor", manual_split=True)
    download_dataset("allenai/wmt22_african", "ibo", "zul", "ibo-zul", manual_split=True)

    download_dataset("allenai/wmt22_african", "kam", "kin", "kam-kin", manual_split=True)
    download_dataset("allenai/wmt22_african", "kam", "lug", "kam-lug", manual_split=True)
    download_dataset("allenai/wmt22_african", "kam", "luo", "kam-luo", manual_split=True)
    download_dataset("allenai/wmt22_african", "kam", "nso", "kam-nso", manual_split=True)
    download_dataset("allenai/wmt22_african", "kam", "nya", "kam-nya", manual_split=True)
    download_dataset("allenai/wmt22_african", "kam", "orm", "kam-orm", manual_split=True)
    download_dataset("allenai/wmt22_african", "kam", "sna", "kam-sna", manual_split=True)
    download_dataset("allenai/wmt22_african", "kam", "som", "kam-som", manual_split=True)
    download_dataset("allenai/wmt22_african", "kam", "ssw", "kam-ssw", manual_split=True)
    download_dataset("allenai/wmt22_african", "kam", "swh", "kam-swh", manual_split=True)
    download_dataset("allenai/wmt22_african", "kam", "tsn", "kam-tsn", manual_split=True)
    download_dataset("allenai/wmt22_african", "kam", "tso", "kam-tso", manual_split=True)
    download_dataset("allenai/wmt22_african", "kam", "umb", "kam-umb", manual_split=True)
    download_dataset("allenai/wmt22_african", "kam", "xho", "kam-xho", manual_split=True)
    download_dataset("allenai/wmt22_african", "kam", "yor", "kam-yor", manual_split=True)
    download_dataset("allenai/wmt22_african", "kam", "zul", "kam-zul", manual_split=True)

    download_dataset("allenai/wmt22_african", "kin", "lug", "kin-lug", manual_split=True)
    download_dataset("allenai/wmt22_african", "kin", "luo", "kin-luo", manual_split=True)
    download_dataset("allenai/wmt22_african", "kin", "nso", "kin-nso", manual_split=True)
    download_dataset("allenai/wmt22_african", "kin", "nya", "kin-nya", manual_split=True)
    download_dataset("allenai/wmt22_african", "kin", "orm", "kin-orm", manual_split=True)
    download_dataset("allenai/wmt22_african", "kin", "sna", "kin-sna", manual_split=True)
    download_dataset("allenai/wmt22_african", "kin", "som", "kin-som", manual_split=True)
    download_dataset("allenai/wmt22_african", "kin", "ssw", "kin-ssw", manual_split=True)
    download_dataset("allenai/wmt22_african", "kin", "swh", "kin-swh", manual_split=True)
    download_dataset("allenai/wmt22_african", "kin", "tsn", "kin-tsn", manual_split=True)
    download_dataset("allenai/wmt22_african", "kin", "tso", "kin-tso", manual_split=True)
    download_dataset("allenai/wmt22_african", "kin", "umb", "kin-umb", manual_split=True)
    download_dataset("allenai/wmt22_african", "kin", "xho", "kin-xho", manual_split=True)
    download_dataset("allenai/wmt22_african", "kin", "yor", "kin-yor", manual_split=True)
    download_dataset("allenai/wmt22_african", "kin", "zul", "kin-zul", manual_split=True)

    download_dataset("allenai/wmt22_african", "lug", "luo", "lug-luo", manual_split=True)
    download_dataset("allenai/wmt22_african", "lug", "nso", "lug-nso", manual_split=True)
    download_dataset("allenai/wmt22_african", "lug", "nya", "lug-nya", manual_split=True)
    download_dataset("allenai/wmt22_african", "lug", "orm", "lug-orm", manual_split=True)
    download_dataset("allenai/wmt22_african", "lug", "sna", "lug-sna", manual_split=True)
    download_dataset("allenai/wmt22_african", "lug", "som", "lug-som", manual_split=True)
    download_dataset("allenai/wmt22_african", "lug", "ssw", "lug-ssw", manual_split=True)
    download_dataset("allenai/wmt22_african", "lug", "swh", "lug-swh", manual_split=True)
    download_dataset("allenai/wmt22_african", "lug", "tsn", "lug-tsn", manual_split=True)
    download_dataset("allenai/wmt22_african", "lug", "tso", "lug-tso", manual_split=True)
    download_dataset("allenai/wmt22_african", "lug", "umb", "lug-umb", manual_split=True)
    download_dataset("allenai/wmt22_african", "lug", "xho", "lug-xho", manual_split=True)
    download_dataset("allenai/wmt22_african", "lug", "yor", "lug-yor", manual_split=True)
    download_dataset("allenai/wmt22_african", "lug", "zul", "lug-zul", manual_split=True)

    download_dataset("allenai/wmt22_african", "luo", "nso", "luo-nso", manual_split=True)
    download_dataset("allenai/wmt22_african", "luo", "nya", "luo-nya", manual_split=True)
    download_dataset("allenai/wmt22_african", "luo", "orm", "luo-orm", manual_split=True)
    download_dataset("allenai/wmt22_african", "luo", "sna", "luo-sna", manual_split=True)
    download_dataset("allenai/wmt22_african", "luo", "som", "luo-som", manual_split=True)
    download_dataset("allenai/wmt22_african", "luo", "ssw", "luo-ssw", manual_split=True)
    download_dataset("allenai/wmt22_african", "luo", "swh", "luo-swh", manual_split=True)
    download_dataset("allenai/wmt22_african", "luo", "tsn", "luo-tsn", manual_split=True)
    download_dataset("allenai/wmt22_african", "luo", "tso", "luo-tso", manual_split=True)
    download_dataset("allenai/wmt22_african", "luo", "umb", "luo-umb", manual_split=True)
    download_dataset("allenai/wmt22_african", "luo", "xho", "luo-xho", manual_split=True)
    download_dataset("allenai/wmt22_african", "luo", "yor", "luo-yor", manual_split=True)
    download_dataset("allenai/wmt22_african", "luo", "zul", "luo-zul", manual_split=True)

    download_dataset("allenai/wmt22_african", "nso", "nya", "nso-nya", manual_split=True)
    download_dataset("allenai/wmt22_african", "nso", "orm", "nso-orm", manual_split=True)
    download_dataset("allenai/wmt22_african", "nso", "sna", "nso-sna", manual_split=True)
    download_dataset("allenai/wmt22_african", "nso", "som", "nso-som", manual_split=True)
    download_dataset("allenai/wmt22_african", "nso", "ssw", "nso-ssw", manual_split=True)
    download_dataset("allenai/wmt22_african", "nso", "swh", "nso-swh", manual_split=True)
    download_dataset("allenai/wmt22_african", "nso", "tsn", "nso-tsn", manual_split=True)
    download_dataset("allenai/wmt22_african", "nso", "tso", "nso-tso", manual_split=True)
    download_dataset("allenai/wmt22_african", "nso", "umb", "nso-umb", manual_split=True)
    download_dataset("allenai/wmt22_african", "nso", "xho", "nso-xho", manual_split=True)
    download_dataset("allenai/wmt22_african", "nso", "yor", "nso-yor", manual_split=True)
    download_dataset("allenai/wmt22_african", "nso", "zul", "nso-zul", manual_split=True)

    download_dataset("allenai/wmt22_african", "nya", "orm", "nya-orm", manual_split=True)
    download_dataset("allenai/wmt22_african", "nya", "sna", "nya-sna", manual_split=True)
    download_dataset("allenai/wmt22_african", "nya", "som", "nya-som", manual_split=True)
    download_dataset("allenai/wmt22_african", "nya", "ssw", "nya-ssw", manual_split=True)
    download_dataset("allenai/wmt22_african", "nya", "swh", "nya-swh", manual_split=True)
    download_dataset("allenai/wmt22_african", "nya", "tsn", "nya-tsn", manual_split=True)
    download_dataset("allenai/wmt22_african", "nya", "tso", "nya-tso", manual_split=True)
    download_dataset("allenai/wmt22_african", "nya", "umb", "nya-umb", manual_split=True)
    download_dataset("allenai/wmt22_african", "nya", "xho", "nya-xho", manual_split=True)
    download_dataset("allenai/wmt22_african", "nya", "yor", "nya-yor", manual_split=True)
    download_dataset("allenai/wmt22_african", "nya", "zul", "nya-zul", manual_split=True)

    download_dataset("allenai/wmt22_african", "orm", "sna", "orm-sna", manual_split=True)
    download_dataset("allenai/wmt22_african", "orm", "som", "orm-som", manual_split=True)
    download_dataset("allenai/wmt22_african", "orm", "ssw", "orm-ssw", manual_split=True)
    download_dataset("allenai/wmt22_african", "orm", "swh", "orm-swh", manual_split=True)
    download_dataset("allenai/wmt22_african", "orm", "tsn", "orm-tsn", manual_split=True)
    download_dataset("allenai/wmt22_african", "orm", "tso", "orm-tso", manual_split=True)
    download_dataset("allenai/wmt22_african", "orm", "umb", "orm-umb", manual_split=True)
    download_dataset("allenai/wmt22_african", "orm", "xho", "orm-xho", manual_split=True)
    download_dataset("allenai/wmt22_african", "orm", "yor", "orm-yor", manual_split=True)
    download_dataset("allenai/wmt22_african", "orm", "zul", "orm-zul", manual_split=True)

    download_dataset("allenai/wmt22_african", "sna", "som", "sna-som", manual_split=True)
    download_dataset("allenai/wmt22_african", "sna", "ssw", "sna-ssw", manual_split=True)
    download_dataset("allenai/wmt22_african", "sna", "swh", "sna-swh", manual_split=True)
    download_dataset("allenai/wmt22_african", "sna", "tsn", "sna-tsn", manual_split=True)
    download_dataset("allenai/wmt22_african", "sna", "tso", "sna-tso", manual_split=True)
    download_dataset("allenai/wmt22_african", "sna", "umb", "sna-umb", manual_split=True)
    download_dataset("allenai/wmt22_african", "sna", "xho", "sna-xho", manual_split=True)
    download_dataset("allenai/wmt22_african", "sna", "yor", "sna-yor", manual_split=True)
    download_dataset("allenai/wmt22_african", "sna", "zul", "sna-zul", manual_split=True)

    download_dataset("allenai/wmt22_african", "som", "ssw", "som-ssw", manual_split=True)
    download_dataset("allenai/wmt22_african", "som", "swh", "som-swh", manual_split=True)
    download_dataset("allenai/wmt22_african", "som", "tsn", "som-tsn", manual_split=True)
    download_dataset("allenai/wmt22_african", "som", "tso", "som-tso", manual_split=True)
    download_dataset("allenai/wmt22_african", "som", "umb", "som-umb", manual_split=True)
    download_dataset("allenai/wmt22_african", "som", "xho", "som-xho", manual_split=True)
    download_dataset("allenai/wmt22_african", "som", "yor", "som-yor", manual_split=True)
    download_dataset("allenai/wmt22_african", "som", "zul", "som-zul", manual_split=True)

    download_dataset("allenai/wmt22_african", "ssw", "swh", "ssw-swh", manual_split=True)
    download_dataset("allenai/wmt22_african", "ssw", "tsn", "ssw-tsn", manual_split=True)
    download_dataset("allenai/wmt22_african", "ssw", "tso", "ssw-tso", manual_split=True)
    download_dataset("allenai/wmt22_african", "ssw", "umb", "ssw-umb", manual_split=True)
    download_dataset("allenai/wmt22_african", "ssw", "xho", "ssw-xho", manual_split=True)
    download_dataset("allenai/wmt22_african", "ssw", "yor", "ssw-yor", manual_split=True)
    download_dataset("allenai/wmt22_african", "ssw", "zul", "ssw-zul", manual_split=True)

    download_dataset("allenai/wmt22_african", "swh", "tsn", "swh-tsn", manual_split=True)
    download_dataset("allenai/wmt22_african", "swh", "tso", "swh-tso", manual_split=True)
    download_dataset("allenai/wmt22_african", "swh", "umb", "swh-umb", manual_split=True)
    download_dataset("allenai/wmt22_african", "swh", "xho", "swh-xho", manual_split=True)
    download_dataset("allenai/wmt22_african", "swh", "yor", "swh-yor", manual_split=True)
    download_dataset("allenai/wmt22_african", "swh", "zul", "swh-zul", manual_split=True)

    download_dataset("allenai/wmt22_african", "tsn", "tso", "tsn-tso", manual_split=True)
    download_dataset("allenai/wmt22_african", "tsn", "umb", "tsn-umb", manual_split=True)
    download_dataset("allenai/wmt22_african", "tsn", "xho", "tsn-xho", manual_split=True)
    download_dataset("allenai/wmt22_african", "tsn", "yor", "tsn-yor", manual_split=True)
    download_dataset("allenai/wmt22_african", "tsn", "zul", "tsn-zul", manual_split=True)

    download_dataset("allenai/wmt22_african", "tso", "umb", "tso-umb", manual_split=True)
    download_dataset("allenai/wmt22_african", "tso", "xho", "tso-xho", manual_split=True)
    download_dataset("allenai/wmt22_african", "tso", "yor", "tso-yor", manual_split=True)
    download_dataset("allenai/wmt22_african", "tso", "zul", "tso-zul", manual_split=True)

    download_dataset("allenai/wmt22_african", "umb", "xho", "umb-xho", manual_split=True)
    download_dataset("allenai/wmt22_african", "umb", "yor", "umb-yor", manual_split=True)
    download_dataset("allenai/wmt22_african", "umb", "zul", "umb-zul", manual_split=True)

    download_dataset("allenai/wmt22_african", "xho", "yor", "xho-yor", manual_split=True)
    download_dataset("allenai/wmt22_african", "xho", "zul", "xho-zul", manual_split=True)

    download_dataset("allenai/wmt22_african", "yor", "zul", "yor-zul", manual_split=True)

    download_dataset("en-vi-translation", "en", "vi", manual_split=True, collation_fn=lambda example: { 'translation': { 'en': example['input'], 'vi': example['output'] } })

def train_tokenizers(vocab_size, output_dir):
    all_training_datafiles = glob.glob('data/train_*')

    lang_abbr_to_files = {}

    for datafile in all_training_datafiles:
        lang_abbreviation = datafile.split('.')[-1]

        if lang_abbreviation in lang_abbr_to_files:
            lang_abbr_to_files[lang_abbreviation].append(datafile)

    for lang, file_list in lang_abbr_to_files.items():
        # preliminary cleanup
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if os.path.exists(f'tokenizer_{lang}.txt'):
            os.remove(f'tokenizer_{lang}.txt')

        # append all files to one file for yttm training, then delete temp file
        with open(f'tokenizer_{lang}.txt', 'w') as outfile:
            for fname in file_list:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)

        # train tokenizer
        yttm.BPE.train(data=f"tokenizer_{lang}.txt", vocab_size=vocab_size, model=os.path.join(output_dir, f"tokenizer_{lang}.model"))

        # remove temp file
        os.remove(f'tokenizer_{lang}.txt')

def prune_data(maxlen):
    for datafile in glob.glob('data/train_*') + glob.glob('data/validation_*'):
        with open(datafile, 'r') as f:
            data = f.readlines()

        with open(datafile, 'w') as f:
            tokenizer = yttm.BPE(f'tokenizers/tokenizer_{datafile.split(".")[-1]}.model')

            for line in data:
                if len(tokenizer.encode(line.strip())) <= maxlen:
                    f.write(line)

if __name__ == '__main__':
    download_base_traindata()

    train_tokenizers(10000, 'tokenizers')

    # prune_data(maxlen=256)
