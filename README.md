# dtu-02456-numpy-ffnn-group27

# NumPy FFNN from Scratch — Group 27 (DTU 02456)

Implementing a fully-connected feedforward neural network **from scratch** in NumPy:
forward/backward propagation, SGD/Adam, L2 regularisation, and experiment tracking with **Weights & Biases**.

## Project structure
src/ # model, training loop, utils
notebooks/ # experiments & gradient checks
data/ # datasets (gitignored)
results/ # plots, confusion matrices
report/ # final 4-page paper

# Setting up your virtual environment
Before running any code, each one of us should create their own local virtual environment.
This keeps dependencies isolated and avoids conflicts between different systems.

For Windows (PowerShell):

        py -m venv .venv
        Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
        .\.venv\Scripts\Activate.ps1


For macOS / Linux:

        python3 -m venv .venv
        source .venv/bin/activate


Once activated, install the shared dependencies:

        pip install -r requirements.txt


Each person uses their own .venv/ (not shared through Git). This ensures everyone works with the same package versions without interfering with global Python installations.