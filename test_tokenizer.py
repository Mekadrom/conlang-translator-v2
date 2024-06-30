from tokenizers import Tokenizer, pre_tokenizers

tokenizer = Tokenizer.from_file("tokenizers/tokenizer_collated.json")

print(type(tokenizer))
print(tokenizer.normalizer)

print("▁Any" in tokenizer.get_vocab())
print("one" in tokenizer.get_vocab())

while True:
    str_in = input("Enter a string: ")

    ids = tokenizer.encode(str_in, add_special_tokens=True).ids

    print(type(ids))
    print(ids)

    decoded = tokenizer.decode(ids, skip_special_tokens=False)

    print(type(decoded))
    print(decoded)
    print(''.join(decoded.split()))
