from itertools import count
from pathlib import Path
import string

import torch

from rnn import RNN


ALPHABET = string.printable
ALPHABET_SIZE = len(ALPHABET) + 1

HIDDEN_SIZE = 16

BATCH_SIZE = 1024
MODEL_SAVE_PATH = Path("./char_rnn.pth")


indices = count()
lookup = {char: next(indices) for char in ALPHABET}
unknown_char_index = next(indices)


def char_to_index(char):
    return lookup.get(char, unknown_char_index)


def one_hot_encode(index):
    zeros = torch.zeros(ALPHABET_SIZE)
    zeros[index] = 1
    return zeros


net = RNN(input_size=ALPHABET_SIZE, hidden_size=HIDDEN_SIZE, output_size=ALPHABET_SIZE)

if MODEL_SAVE_PATH.exists():
    print("Loading trained model from file")
    net.load_state_dict(torch.load(MODEL_SAVE_PATH))
else:
    raise Exception("Please train RNN first")

net.eval()

seed = "What do you believe is going to happen?"

chars = seed
chars_tensor = torch.stack(tuple(map(one_hot_encode, map(char_to_index, chars))))
torch.manual_seed(2)
hidden_state = torch.randn(HIDDEN_SIZE)
pred_chars, hidden_state = net(chars_tensor, hidden_state)


pred_chars_indices = pred_chars.argmax(axis=-1)
pred_chars = [ALPHABET[idx] for idx in pred_chars_indices]
print("".join(pred_chars))
