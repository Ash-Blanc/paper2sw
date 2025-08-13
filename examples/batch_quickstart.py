from __future__ import annotations

from paper2sw import Predictor


def main() -> None:
    predictor = Predictor.from_pretrained(enable_cache=True, selection_keep_ratio=0.5)
    inputs = [
        "https://arxiv.org/abs/2411.07191",
        "../README.md",
        "../LICENSE",
    ]
    results = predictor.predict_batch(inputs, top_k=3, seed=123)
    for i, preds in enumerate(results):
        print(f"Input {i} -> {len(preds)} predictions")
        for p in preds:
            print(p.to_dict())


if __name__ == "__main__":
    main()