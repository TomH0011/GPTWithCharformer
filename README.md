# GPT with Charformer Tokenisation

This project is a demonstration of a GPT-style language model that uses **Charformer** for subword tokenisation. Instead of a static vocabulary (like WordPiece or BPE), it learns to group characters into meaningful tokens directly from the data.

As well as this it has the option for a simpler transofrmer for character level encoding


## How It Works

1.  **Tokenisation (Charformer):** The input text is processed by Gradient-based Subword Tokenisation (GBST) blocks. These blocks downsample the character sequence, effectively "learning" to group characters into subword units.
2.  **Language Modeling:** The sequence of learned tokens is then fed into a standard Transformer Decoder architecture. Using causal self-attention, the model learns to predict the next token in a sequence.
3.  **Generation:** After training, the model can generate new text by starting with a prompt and iteratively predicting the next token.

## Tech Stack

* Python
* PyTorch
* numpy

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

2.  **Tidy data:** Normalise any white spaces, clean the data so it has context and its all utf-8 characters

3.  **Configure Training:** Adjust hyperparameters like `max_iters`, `batch_size`, and model dimensions in the `Config.py` file.

4.  **Run the Script:**
    ```bash
    python main.py
    ```
    * The first time you run the script, it will train the model from scratch and save the weights to `charformer_checkpoint_{n}.pth`. Where n is a multiple of 5000 up to max_iterations
    * On subsequent runs, it will automatically load the saved weights and proceed directly to text generation. To force retraining, delete the previous saved model files.

## Acknowledgements

* Google Deepminds Charformer paper arXiv:2106.12672 [cs.CL] foiund here: (https://arxiv.org/abs/2106.12672)
* The Charformer module is based on the excellent work by Phil Wang (lucidrains). You can find his repository here: [charformer-pytorch](https://github.com/lucidrains/charformer-pytorch).
