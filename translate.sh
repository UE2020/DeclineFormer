#!/usr/bin/env bash
./target/release/seq2seq test model_125.pt "$(cd ../DeclEngine && python3 test.py "$1")"
