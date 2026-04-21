from tokenizer.tokenizer import tokenize_code, build_vocab
from utils.dataset import create_dataset
from model.model import IntentModel
from trainer.train import train
from inference.predict import predict

# sample labeled intents
samples = [
    ("for i in range ( 10 ) :", 0),  # loop
    ("if x > 10 :", 1),             # condition
    ("def add a b :", 2),           # function
]

# tokenize all
all_tokens = []
for code, _ in samples:
    all_tokens.extend(tokenize_code(code))

word2idx, idx2word = build_vocab(all_tokens)

X, y = create_dataset(samples, word2idx)

model = IntentModel(len(word2idx), num_classes=3)

train(model, X, y)

# test
test_seq = X[0]
print("Predicted intent:", predict(model, test_seq))
