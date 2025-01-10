# Hindi BPE Tokenizer

This repository contains an implementation of a Byte Pair Encoding (BPE) tokenizer specifically tailored for the Hindi language. It uses the Hindi Unicode range to tokenize text while supporting spaces and basic punctuation.

## Features

- **Customizable Vocabulary Size:** Specify the desired vocabulary size for the tokenizer.
- **Special Token Support:** Handles `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`, and spaces explicitly.
- **BPE Training:** Trains on Hindi datasets to generate a vocabulary of subword units.
- **Text Cleaning:** Preserves Hindi characters and basic punctuation while cleaning unnecessary symbols.
- **Encoding and Decoding:** Converts text to token IDs and reconstructs text from token IDs.
- **Save and Load:** Saves and loads tokenizer configurations, including vocabulary and merge rules.

---

## DEMO 

The app is hosted on huggingface spaces [here](https://huggingface.co/spaces/wgetdd/Hindi_BPE)

## Usage

### Training the Tokenizer
Train the tokenizer on your dataset:

```python
from tokenizer import HindiBPETokenizer, prepare_dataset

dataset_iterator = prepare_dataset(dataset_path)

tokenizer = HindiBPETokenizer(vocab_size=5000)
tokenizer.train(dataset_iterator)

# Save the trained tokenizer
tokenizer.save_tokenizer("hindi_tokenizer_v1")
```

---

### Encoding and Decoding
Once trained, use the tokenizer to encode and decode text:

```python
test_text = "नमस्ते दुनिया"
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)

print(f"Original: {test_text}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
```

---

### Loading the Tokenizer
Load a saved tokenizer for reuse:

```python
loaded_tokenizer = HindiBPETokenizer.load_tokenizer("hindi_tokenizer_v1")

# Verify encoding and decoding
encoded_loaded = loaded_tokenizer.encode("नमस्ते दुनिया")
decoded_loaded = loaded_tokenizer.decode(encoded_loaded)
print(f"Decoded (loaded): {decoded_loaded}")
```

---

## Gradio Demo
A **Gradio** web application is included to interact with the tokenizer. 

#### Run the `gradio_app.py` file locally:
    ```bash
    python app.py
    ```

The Hugging Face Space configuration should point to the `gradio_app.py` entry point.

---

## Example Dataset
Use the Hindi summarization corpus available on [Kaggle](https://www.kaggle.com/datasets/disisbig/hindi-text-short-and-large-summarization-corpus). Ensure the `train.csv` and `test.csv` files are structured correctly for the `prepare_dataset` function.

Dataset Path:
```bash
/kaggle/input/hindi-text-short-and-large-summarization-corpus/
```

---

## File Structure
```
.
├── train.py           # Core tokenizer implementation
├── app.py             # Gradio demo for interactive tokenization
├── README.md          # Documentation
└── hindi_tokenizer_v4 # Saved tokenizer configurations
```

---

