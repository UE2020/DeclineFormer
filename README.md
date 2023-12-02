# DeclineFormer
Translating Latin to English using a sequence-to-sequence transformer augmented with learnable morphologically-derived grammatical embeddings.

## Usage
### Evaluation
Evaluation requires two steps.

First, [DeclEngine should be cloned and compiled](https://github.com/BlueCannonBall/DeclEngine).

A sentence should be translated to IR using the `test.py` script in DeclEngine:
```
$ python3 test.py "Qui Deum non audiunt, certe peribunt."
how<SEP><ACC>God<SEP>not<SEP>they hear,<SEP>surely<SEP>they will die.
```

Then the string should be used with the `test` subcommand:
```
$ ./target/release/seq2seq test model.pt "how<SEP><ACC>God<SEP>not<SEP>they hear,<SEP>surely<SEP>they will die."
output: <BOS> those who do not listen to god, will they die.<EOS>
```

Well done! Comparing Google Translate and our result:

| | Translation |
|---|---|
| **GTranslate** | *Those who do not listen to God will surely perish.* |
| **Our result** | *those who do not listen to god, will they die.* |

A comparison of translations from the Latin Vulgate (Genesis 8:7):

| | Translation |
|---|---|
| **Latin** | *qui egrediebatur, et non revertebatur, donec siccarentur aquae super terram.* |
| **Ground Truth** | *which went forth and did not return, until the waters were dried up across the earth.* |
| **GTranslate** | *who went out and did not return until the waters were dried up on the earth.* |
| **Our result** | *and he went out, and he did not return, until the waters were dried up upon the earth.* |

### Tokenization

Tokenizers can be tested using the `test-tok` command:
```sh
$ ./target/release/seq2seq test-tok tokenizer.json "This is a test." # <tokenizer> <test-sentence>
["Ġthis", "Ġis", "Ġa", "Ġtest", "."]
```
### Training
Training (tensorboard logs are written to `./logdir/train`):
```sh
$ python3 src/model.py # generate torchscripts
$ python3 src/split.py # split data
$ ./target/release/seq2seq train ir.txt en.txt 5000 ir-en.txt false 1 # last parameter is number of hours before quitting
Epoch 1 complete!
Epoch 2 complete!
...
Epoch 18 complete!
```
Checkpoints will be saved to `model_<EPOCH>.pt`
