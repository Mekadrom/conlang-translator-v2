from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizers/tokenizer_collated.json")

print(type(tokenizer))

while True:
    str_in = input("Enter a string: ")

    ids = tokenizer.encode(str_in).ids

    print(type(ids))
    print(ids)

    decoded = tokenizer.decode(ids, skip_special_tokens=True)

    print(type(decoded))
    print(decoded)
    print(''.join(decoded.split()))
