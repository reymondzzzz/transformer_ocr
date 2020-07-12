input_size = (96, 96)
inner_size = (256, 256)
augment_dropout = 0.1
empty_dropout = 0.1
batch_size = 64
epochs = 10000
label_len = 36
# vocab = "0123456789abcdefghijklmnopqrstuvwxyzäüö"
vocab =  "<0123456789abcdefghijklmnopqrstuvwxyzäüö>"
PAD = 'PAD'
# SOS = 'SOS'
# EOS = 'EOS'
char2token = {
    "PAD": 0,
    # "SOS": 1,
    # "EOS": 2,
}
token2char = {
    0: "PAD",
    # 1: "SOS",
    # 2: "EOS",
}
for i, c in enumerate(vocab):
    # char2token[c] = i + 3
    # token2char[i + 3] = c
    char2token[c] = i + 1
    token2char[i + 1] = c

data_config = {
    'root_path': '/home/kstarkov/ml/datasets/ocr/lpr4_images',
    'lines_allowed': [1, 2],
}

ckpt_dir = '/home/kstarkov/work2/checkpoints/ocr'

fake_generator_config = {
    'lpr_resources': '/home/kstarkov/t1s/tech1lpr/lpr_resources',
    'most_popular_templates': {
        'ru_type5_subtype1_lines1': 0.033,
        'ru_type5_subtype2_lines1': 0.033,
        'ru_type5_subtype3_lines1': 0.033,
        'ru_type6_subtype1_lines1': 0.1,
        'ru_type7_subtype1_lines1': 0.1,
    }
}
