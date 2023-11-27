# DeclineFormer
Translating Latin to English using a sequence-to-sequence transformer augmented with domain knowledge.

## Usage
Evaluation:
```sh
$ ./target/release/seq2seq test model.pt "<NOM>this<SEP><ABL>gate<SEP>just,<SEP><GEN>of owner<SEP><ACT>they will enter<SEP>in<SEP><ACT>may I go."
output: <BOS> this is the gate of the lord. the just will enter by it.<EOS>
```

Tokenization:
```sh
$ ./target/release/seq2seq test-tok bcb-en.txt 5000 "This is a test." # <tokenization-data> <vocab-size> <test-sentence>
["Ġthis", "Ġis", "Ġa", "Ġtest", "."]
```

Training (tensorboard logs are written to `./logdir/train`):
```sh
$ python3 src/model.py # generate torchscripts
$ ./target/release/seq2seq train bcb-en.txt 5000 bcb-en.txt # <tokenization-data> <vocab-size> <training-sentence>
Epoch 1 complete!
Epoch 2 complete!
...
Epoch 18 complete!
```
Checkpoints will be saved to `model-<EPOCH>.pt`
