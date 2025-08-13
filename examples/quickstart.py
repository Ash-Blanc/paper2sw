from __future__ import annotations

from paper2sw import predict_super_weights, Predictor


def main() -> None:
    # Minimal usage
    predictions = predict_super_weights(
        paper="https://arxiv.org/abs/2411.07191",
        top_k=5,
        use_openai_fallback=True,
    )

    for p in predictions:
        print(p.to_dict())

    # Slightly more control
    predictor = Predictor.from_pretrained(
        model_id="paper2sw/paper2sw-diff-base",
        device="cpu",
        precision="bf16",
    )

    predictions = predictor.predict(
        paper="./README.md",
        top_k=3,
        seed=42,
    )

    predictor.save_jsonl(predictions, path="./sw.jsonl")


if __name__ == "__main__":
    main()