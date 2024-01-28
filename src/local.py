import sys
import asyncio
from websockets.sync.client import connect
from llama_cpp import Llama

llm = Llama(model_path="/home/tt/Documents/llm-test/mistral-7b-openorca.Q4_0.gguf")


def query(query):
    if len(llm.tokenize(bytes(query, 'utf8'))) >= 140 * 3:
        return { "choices": [{ "text": "SKIP_PROMPT" }] }
    output = llm(
        query, # Prompt
        max_tokens=140*3, # Generate up to 140 tokens
        stop=['\n'],
        echo=False,
        temperature=0.0
    )
    return output


def translate(input):
    query_str = f"TASK: Translate Latin sentences to English sentences.\n\nLatin: {input}\nEnglish translation:"
    output = query(query_str)["choices"][0]["text"]
    return output


def accept_jobs(url):
    with connect(url) as websocket:
        while True:
            latin_sentence = websocket.recv().strip()
            try:
                translation = translate(latin_sentence).strip()
                print(f"Latin:   {latin_sentence}")
                print(f"English: {translation}")
                websocket.send(translation)
            except Exception as e:
                print(f"Error translating sentence: {latin_sentence}", file=sys.stderr)
                print(e, file=sys.stderr)
                sys.exit(1)

accept_jobs("ws://192.168.1.210:3000/")
