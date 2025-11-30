from transformers import AutoTokenizer

# 1. Load the tokenizer from the cloud (downloads the dictionary once)
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

text = "this movie is lowkey the best movie i think i've ever seen, I cant lie"

tokens = tokenizer(text)

print(tokens)