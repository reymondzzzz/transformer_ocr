def tokenize_vocab(vocab):
    letter_to_token, token_to_letter = {}, {}
    for idx, letter in enumerate(vocab):
        letter_to_token[letter] = idx
        token_to_letter[idx] = letter
    return letter_to_token, token_to_letter
