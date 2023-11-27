# DeclineFormer
Translating Latin to English using a sequence-to-sequence transformer augmented with domain knowledge.

## Usage
Evaluation:
```sh
$ ./target/release/seq2seq test model.pt "<NOM>who<SEP>it was going,<SEP>and<SEP>not<SEP>it was turn backed,<SEP>while<SEP>they would be dried<SEP><GEN>of water<SEP>above<SEP><ACC>earth."
output: <BOS> and so he went forth and did not return, until the water of the earth were dried up upon the earth.<EOS>
```

The above is a translation of the Latin Vulgate verse of Genesis 8:7, after training 16 epochs on the included annotated dataset
```
qui egrediebatur, et non revertebatur, donec siccarentur aquae super terram.
which went forth and did not return, until the waters were dried up across the earth.
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
