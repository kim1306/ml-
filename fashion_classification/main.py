"""Run the full pipeline for one or all models:
tune with Optuna -> train the best config -> evaluate -> visualize -> summary.

Usage:
    python main.py --data-root ./data --model all
    python main.py --data-root ./data --model mlp --n-trials 10

TensorBoard curves are written to runs/<model>; view them with:
    tensorboard --logdir runs
"""
import argparse

import pandas as pd

from fashion_clf import (DataPaths, get_device, set_seed, tune_mlp, tune_pretrained_vgg,
                         tune_vgg_scratch, train_final_mlp, train_final_pretrained_vgg,
                         train_final_vgg_scratch)

# name -> (title, tuning function, final-training function)
PIPELINES = {
    "mlp": ("MLP", tune_mlp, train_final_mlp),
    "vgg_scratch": ("VGG16 (scratch)", tune_vgg_scratch, train_final_vgg_scratch),
    "vgg_pretrained": ("VGG16 (pretrained)", tune_pretrained_vgg, train_final_pretrained_vgg),
}


def run_model(name, paths, device, seed, n_trials=None):
    title, tune_fn, train_fn = PIPELINES[name]

    # 1. hyperparameter search (Optuna); n_trials=None keeps each model's default
    tune_kwargs = {} if n_trials is None else {"n_trials": n_trials}
    study = tune_fn(paths, device, seed=seed, **tune_kwargs)
    print(f"Best {title} val_acc: {study.best_value:.4f}")
    print(f"Best {title} params:", study.best_params)

    # 2. final training with the best configuration (logs to runs/<name>)
    best_cfg = study.best_trial.user_attrs["config"]
    print(f"Best {title} config:", best_cfg)
    dm, trainer, history, best_val_acc = train_fn(best_cfg, paths, device, log_dir=f"runs/{name}")

    # 3. evaluation, sample predictions, unlabeled test predictions
    val_acc, mean_ap = trainer.full_evaluation(dm.valid_loader(), title)
    trainer.visualize_predictions(dm.valid_dataset, n=8, title=f"{title} — sample validation predictions")
    test_loader, test_dataset = dm.test_loader()
    trainer.predict_test_set(test_loader, test_dataset, title=title)

    return {"model": title, "val_accuracy": val_acc, "mAP": mean_ap, "best_config": best_cfg}


def main():
    parser = argparse.ArgumentParser(description="Fashion item classification")
    parser.add_argument("--data-root", default="./data",
                        help="folder containing train/, valid/ and test/")
    parser.add_argument("--model", default="all", choices=["all", *PIPELINES])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=None,
                        help="override the number of Optuna trials (default: per-model value)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("Using device:", device)
    paths = DataPaths.from_root(args.data_root)

    names = list(PIPELINES) if args.model == "all" else [args.model]
    results = [run_model(n, paths, device, args.seed, args.n_trials) for n in names]

    pd.set_option("display.max_colwidth", None)
    print("\nSummary")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
