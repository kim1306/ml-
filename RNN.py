"""
LSTM vs Transformer: running-sum task (mod 20)

Compares an LSTM encoder-decoder and a Transformer encoder-decoder on predicting
the running sum of a random sequence, with cumulative sums wrapped into 1..20.
Example: input 4, 9, 15, 2 -> output 4, 13, 8, 10.

Both models are kept under a similar parameter budget (MAX_PARAMS), and
hyperparameters are optionally tuned with Optuna.

Requirements:
    pip install torch matplotlib optuna torchinfo
"""


# ======================================================================
# 1. Imports and settings
# ======================================================================
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import optuna

TASK = "running_sum" # other task modes may be developed in the future
VOCAB =20 # tokens are the numbers 0..19
BOS =0 # special "begin of sequence" token has id 0
NUM_IDS =VOCAB + 1 # ids 0..20 (BOS + 20 real tokens) = 21 embeddings and outputs

D_MODEL =64  # size of every token's vector, it is for LSTM (not optimized with Optuna)
# because keep it fixed simplifies the search for hidden state's size
# finetuning hyperparameters for transformer heavily influence the accuracy of transformer, so it gets optimized for transformer
LSTM_HIDDEN =64 #a vector that carries information from the current time(short-term)
N_HEADS =8 # attention heads in the transformer
# dimmensions (64) gets divided fairly to each head (8)
N_LAYERS =2 # layers in the transformer encoder and decoder
# it stacks two LSTM layers vertically, so the second layer takes the first one's hidden output as input
# it creates two encoder and 2 decoder layers in transformer
LR = 1e-3
BATCH_SIZE = 64
FF_SIZE =128
            # FF= an expansion technique used to double the token's vectors size
            # in transformer, it is used when attention does the comparing, FF does the processinng
            # by transform the features and store patterns
            # it shrinks back to the origianl value before passing on to the next layer
SEQ_LENGTHS = [5, 10, 15, 20] # max length
TRAIN_STEPS = 6000
VAL_EVERY= 500
N_VAL= 500
CURRICULUM_EVERY= 100  # add one longer sequence length every this many steps (see train)

RUN_TUNING = True
N_TRIALS = 20 # trials per model
TUNE_STEPS = 3000 # training steps per trial
TUNE_LENGTH = 10 # Optuna trainings are scored on sequences of this length
MAX_PARAMS = 250_000  # bigger models are skipped


torch.manual_seed(0)
random.seed(0)


# ======================================================================
# 2. Data
# ======================================================================
def compute_target(src):
    #The correct answer for a batch of input sequences (shape: batch x length).
    if TASK == "running_sum":
        return (src.cumsum(dim=1) - 1) % VOCAB + 1
    else:
        raise ValueError("task must be 'running_sum', no other task mode is implemented yet")

def make_batch(batch_size, length):
    src = torch.randint(1, VOCAB + 1, (batch_size, length))  # size = 2D, (batch size, length)
    target = compute_target(src)

    # BOS is the toke that informs the model to start decoder
    bos_column = torch.full((batch_size, 1), BOS) # tensor with the size (batch,1) filled with BOS(0)
    decoder_input = torch.cat([bos_column, target[:, :-1]], dim=1)
    # cut off the last token and add BOS at the front, so if input to decoder is there,
    # it calculate 0 + the first token, then first token + second token ...
    return src, decoder_input, target

def make_fixed_set(n_samples, length, seed):
    #different seeds give different sets, so validation and test never overlap.
    generator = torch.Generator().manual_seed(seed)
    src = torch.randint(1, VOCAB + 1, (n_samples, length), generator=generator)
    return src, compute_target(src)



# ======================================================================
# 3. Model 1: LSTM encoder-decoder
# ======================================================================
class LSTMSeq2Seq(nn.Module):
    def __init__(self,d_model=D_MODEL, hidden=LSTM_HIDDEN, layers=N_LAYERS):
        super().__init__()
        #nn.Embedding expands integerIDs' shape by adding the dimension(D_MODEL)
        self.embed = nn.Embedding(NUM_IDS, d_model)
        """
        nn.LSTM helps a model to learn patterns and relationships in sequential data by
        automatically creates internal weights and bias required for forget, input and output gates + candidate cell state
        parameters:
        input = token vector, hidden size=LSTM hidden, number of LSTM layer,
        setting batch_first=True flips that so the batch size comes first.
        Default: The shape is (seq_length, batch_size, embedding_dim)
        batch_first=True -> makes the batch_size comes first
        if not swapping their position, i need to permute the tensor everytime feeding data

       the output of nn.LSTM = output(batch size, sequence length, LSTM HIDDEN) and state(h_n (final hidden state) and c_n(final cell state))
        """
        self.encoder = nn.LSTM(d_model, hidden, num_layers=layers, batch_first=True)
        self.decoder = nn.LSTM(d_model, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, NUM_IDS) # weights between input tokkens and the short-term memory

    def forward(self, src, decoder_input):
#embed(src) means find vectors for all the integers inside this tensor
# after Embedding (self.embed(src)): Shape becomes (batch_size, length, D_MODEL)
# and here is the tensor in the dimension (batch_size, sqe_length, embedding_dim)
        _, state = self.encoder(self.embed(src))
        # nn.LSTM also takes an existing hidden state, it is very important to pass them
        # otherwise decoder starts with its h and c states to all zero by default
        dec_out, _ = self.decoder(self.embed(decoder_input), state)
        return self.out(dec_out) # nn.linear maps the LSTM hidden to NUM_IDS (batch, length, NUM_IDS)

    @torch.no_grad()
    def generate(self, src):
        #Write the output one token at a time, feeding back its own guesses.
        _, state = self.encoder(self.embed(src))
        token = torch.full((src.size(0), 1), BOS)# src.size(0)= batch size
        outputs = []
        for _ in range(src.size(1)):
            dec_out, state = self.decoder(self.embed(token), state)
            token = self.out(dec_out).argmax(dim=-1) # most likely next token
            outputs.append(token)
        return torch.cat(outputs, dim=1)



# ======================================================================
# 4. Model 2: Transformer encoder-decoder
# ======================================================================
class TransformerSeq2Seq(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, layers=N_LAYERS,ff=FF_SIZE,max_len=100):
        super().__init__()
        self.embed = nn.Embedding(NUM_IDS, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        # nn.transformer
        """it takes these parameters and does:
        in encoder- it applies self-attention, it feedforward and send Keys and Values
        in decoder - it applies masked self attention, and cross attention, and then feedforward, then send Qeury
        then it automatically does the matrix on Q and K to get attention scores
        the result then get multiplied by V matrix, these things all automatically happens """
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=layers,
            num_decoder_layers=layers,
            dim_feedforward=ff,
            dropout=0.0,
            batch_first=True,
        )
        self.out = nn.Linear(d_model, NUM_IDS)

    def add_positions(self, tokens):
        positions = torch.arange(tokens.size(1)).unsqueeze(0)  # unsqeeuze make it 2D: [[0, 1, ...]]
        return self.embed(tokens) + self.pos_embed(positions)

    def forward(self, src, decoder_input):
        length = decoder_input.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(length)
        # generate_square_subsequent_mask generates a square matrix
        # future-token masking to prevent the model to see the (future) input

        hidden = self.transformer(
            # the encoder input and decoder input
            self.add_positions(src),
            self.add_positions(decoder_input),
            tgt_mask=mask, #tgt_mask =target mask, it belong to PyTorch
        )
        # the output of nn.transformer = (batch size, sequence length, D MODEL)
        return self.out(hidden)

    @torch.no_grad()
    def generate(self, src):
        tokens = torch.full((src.size(0), 1), BOS)
        for _ in range(src.size(1)):
            logits = self.forward(src, tokens)
            next_token = logits.select(dim=1, index=-1).argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens[:, 1:] # drop the BOS



# ======================================================================
# 5. Training and evaluation
# ======================================================================
def evaluate(model, src, target):
#eval() and train() switching happens every VAL_EVERY steps

    model.eval()
    prediction = model.generate(src)
    correct = (prediction == target).float() # 1 where the token was right
    model.train()
    return correct.mean(dim=0).tolist()


def average_accuracy(model, datasets): #datasets= 256 validation seq
    #Overall % correct, averaged over several {length: (src, target)} datasets
    scores = []
    for src, target in datasets.values():
        scores.append(sum(evaluate(model, src, target)) / src.size(1))
    return 100 * sum(scores) / len(scores)


def train(model, lengths, val_sets, steps=TRAIN_STEPS,lr=LR, batch_size=BATCH_SIZE, verbose=True ):
    """
    lengths : list of sequence lengths to train on (one is picked at random each step)
    val_sets: {length: (src, target)}, fixed data we check every VAL_EVERY steps
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    for step in range(steps):
        n_allowed = min(len(lengths), 4 + step // CURRICULUM_EVERY)
        length = random.choice(lengths[:n_allowed])
        src, decoder_input, target = make_batch(batch_size, length)
        logits = model(src, decoder_input)
        # CrossEntropyLoss wants (N, classes), so flatten batch and length together
        loss = loss_fn(logits.reshape(-1, NUM_IDS), target.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose and (step + 1) % VAL_EVERY == 0:
            val_acc = average_accuracy(model, val_sets)
            print(f"    step {step + 1:5d}  train loss {loss.item():.3f}  |  validation accuracy {val_acc:5.1f}%")
    return average_accuracy(model, val_sets)


def count_params(model):
    return sum(p.numel() for p in model.parameters())



# ======================================================================
# 6. Hyperparameter tuning with Optuna
# ======================================================================
# Hyperparameter search with Optuna
def make_objective(name):
    lengths = list(range(1, TUNE_LENGTH + 1))
    val_sets = {TUNE_LENGTH: make_fixed_set(N_VAL, TUNE_LENGTH, seed=1)}

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True) # log scale: tries 1e-4, 1e-3, 1e-2 evenly
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        if name == "LSTM":
            model_args = {
                "hidden": trial.suggest_int("hidden", 32, 128, step=16),
                "layers": trial.suggest_int("layers", 1, 2),
            }
            model_class = LSTMSeq2Seq
        else:
            model_args = {
                "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
                "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]), # all should be able to divide every d_model candidates above
                "layers": trial.suggest_int("layers", 1, 3),
                "ff": trial.suggest_categorical("ff", [64, 128, 256]),
            }
            model_class = TransformerSeq2Seq

        torch.manual_seed(trial.number)
        random.seed(trial.number)
        model = model_class(**model_args) # The double asterisk unpacks the dictionary as keyword arguments
        if count_params(model) > MAX_PARAMS:
            raise optuna.TrialPruned()  #prune if the parameters gets too big
        return train(model, lengths, val_sets, steps=TUNE_STEPS,
                     lr=lr, batch_size=batch_size, verbose=False)

    return objective


def tune(name):
    print(f"\ntuning {name}: {N_TRIALS} trials of {TUNE_STEPS} steps each")
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(make_objective(name), n_trials=N_TRIALS)
    print(f"  best validation accuracy: {study.best_value:.1f}%")
    print(f"  best settings: {study.best_params}")
    return study.best_params


def config_from_params(name, params):
    params = dict(params)
    lr = params.pop("lr")
    batch_size = params.pop("batch_size")
    model_class = LSTMSeq2Seq if name == "LSTM" else TransformerSeq2Seq
    return {"model_class": model_class, "model_args": params, "lr": lr, "batch_size": batch_size}



# ======================================================================
# 7. Experiment, table and plot
# ======================================================================
# running the experiment
# One config per model: which class, which model settings, and how to train it.
DEFAULT_CONFIGS = {
    "LSTM":        {"model_class": LSTMSeq2Seq,        "model_args": {}, "lr": LR, "batch_size": BATCH_SIZE},
    "Transformer": {"model_class": TransformerSeq2Seq, "model_args": {}, "lr": LR, "batch_size": BATCH_SIZE},
}


def run_experiment(configs):
    results = {name: {} for name in configs}

    for length in SEQ_LENGTHS:
        val_sets = {length: make_fixed_set(N_VAL, length, seed=1)} # checked during training
        test_src, test_target = make_fixed_set(N_VAL, length, seed=2) # only used once, at the end
        for name, config in configs.items():
            model = config["model_class"](**config["model_args"])
            print(f"\n[{name}]  sequence length {length}  ({count_params(model):,} parameters)")
            train(model, list(range(1, length + 1)), val_sets, lr=config["lr"], batch_size=config["batch_size"])
            results[name][length] = evaluate(model, test_src, test_target)
            print(f"    TEST accuracy: {100 * sum(results[name][length]) / length:.1f}%")
    return results


def print_table(results):
    names = list(results)
    width = 10 + 13 * len(names)
    print("\n" + "=" * width)
    print(f"Test accuracy: average % of tokens correct  (task: {TASK})")
    print("=" * width)
    print(f"{'length':>8} | " + " | ".join(f"{n:>10}" for n in names))
    for length in SEQ_LENGTHS:
        row = [100 * sum(results[n][length]) / length for n in names]
        print(f"{length:>8} | " + " | ".join(f"{a:>9.1f}%" for a in row))


def plot_results(results, filename="memory_compare.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: overall accuracy as the sequence gets longer
    for name in results:
        avg = [100 * sum(results[name][L]) / L for L in SEQ_LENGTHS]
        axes[0].plot(SEQ_LENGTHS, avg, marker="o", label=name)
    axes[0].set_xlabel("sequence length")
    axes[0].set_ylabel("tokens correct (%)")
    axes[0].set_title("Test accuracy vs. sequence length")
    axes[0].legend()

    # Right: WHERE in the sequence does each model go wrong? (longest length)
    longest = SEQ_LENGTHS[-1]
    for name in results:
        axes[1].plot([100 * a for a in results[name][longest]], label=name)
    axes[1].set_xlabel("token position in the sequence")
    axes[1].set_ylabel("tokens correct (%)")
    axes[1].set_title(f"Accuracy by position (length {longest})")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    print(f"\nSaved plot to {filename}")



def show_architecture_diagrams():
    """Display the architecture diagrams if the PNG files exist (notebook only)."""
    import os
    try:
        from IPython.display import Image, display
    except ImportError:
        return
    for f in ("architecture_lstm.png", "architecture_transformer.png"):
        if os.path.exists(f):
            display(Image(filename=f))


def main():
    configs = DEFAULT_CONFIGS
    if RUN_TUNING:
        configs = {name: config_from_params(name, tune(name)) for name in DEFAULT_CONFIGS}

    results = run_experiment(configs)
    print_table(results)
    plot_results(results)



if __name__ == "__main__":
    main()