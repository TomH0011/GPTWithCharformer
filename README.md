# GPT with Charformer Tokenization

This project is a demonstration of a GPT-style language model that uses **Charformer** for subword tokenization. Instead of a static vocabulary (like WordPiece or BPE), it learns to group characters into meaningful tokens directly from the data.

The model is a decoder-only Transformer designed for text generation.


## How It Works

1.  **Tokenization (Charformer):** The input text is processed by Gradient-based Subword Tokenization (GBST) blocks. These blocks downsample the character sequence, effectively "learning" to group characters into subword units.
2.  **Language Modeling:** The sequence of learned tokens is then fed into a standard Transformer Decoder architecture. Using causal self-attention, the model learns to predict the next token in a sequence.
3.  **Generation:** After training, the model can generate new text by starting with a prompt and iteratively predicting the next token.

## Tech Stack

* Python
* PyTorch

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/TomH0011/GPTWithCharformer](https://github.com/TomH0011/GPTWithCharformer)
    cd GPTWithCharformer
    ```

2.  Install the required packages. It is highly recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Prepare Data:** Place your training data in a file named `Sample_Text.txt` in the root directory.

2.  **Configure Training:** Adjust hyperparameters like `max_iters`, `batch_size`, and model dimensions in the `Config.py` file.

3.  **Run the Script:**
    ```bash
    python main.py
    ```
    * The first time you run the script, it will train the model from scratch and save the weights to `charformer_model.pth`.
    * On subsequent runs, it will automatically load the saved weights and proceed directly to text generation. To force retraining, delete the `charformer_model.pth` file.

## Acknowledgements

* This implementation is heavily inspired by Andrej Karpathy's "makemore" series.
* The Charformer module is based on the excellent work by Phil Wang (lucidrains). You can find his repository here: [charformer-pytorch](https://github.com/lucidrains/charformer-pytorch).