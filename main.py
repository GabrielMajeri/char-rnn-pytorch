from itertools import count
import string

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from rnn import RNN


ALPHABET = string.printable
ALPHABET_SIZE = len(ALPHABET) + 1

HIDDEN_SIZE = 16

BATCH_SIZE = 1024

indices = count()
lookup = {char: next(indices) for char in ALPHABET}
unknown_char_index = next(indices)


def char_to_index(char):
    return lookup.get(char, unknown_char_index)


def one_hot_encode(index):
    zeros = torch.zeros(ALPHABET_SIZE)
    zeros[index] = 1
    return zeros


try:
    with open("input/warpeace_input.txt") as fin:
        input_text = fin.read()
except FileNotFoundError as e:
    print(e)
    print("Please download the input data from 'https://cs.stanford.edu/people/karpathy/char-rnn/'")
    print("Place it in the 'input' directory")
    exit(1)

input_length = len(input_text)


net = RNN(input_size=ALPHABET_SIZE,
          hidden_size=HIDDEN_SIZE,
          output_size=ALPHABET_SIZE)

optimizer = optim.RMSprop(net.parameters())

hidden_state = torch.zeros(BATCH_SIZE, HIDDEN_SIZE)

total_loss = 0.0

print("Starting to train char RNN")
i = 0
last_print = 0
while i < input_length:
    if i + BATCH_SIZE >= input_length:
        # TODO: pad last batch to `BATCH_SIZE`
        break

    start, end = i, i + BATCH_SIZE
    chars = input_text[start:end]
    next_chars = input_text[start + 1:end + 1]

    chars_tensor = torch.stack(
        tuple(map(one_hot_encode, map(char_to_index, chars))))
    next_chars_index = tuple(map(char_to_index, next_chars))
    next_chars_tensor = torch.tensor(next_chars_index)

    optimizer.zero_grad()

    pred_chars, hidden_state = net(chars_tensor, hidden_state)

    loss = F.cross_entropy(pred_chars, next_chars_tensor)
    loss.backward()

    loss = loss.item()
    total_loss += loss

    if i - last_print > 1000:
        print(total_loss / i)
        last_print = i

    optimizer.step()

    hidden_state = hidden_state.detach()

    i += BATCH_SIZE
