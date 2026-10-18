"""Inspect the SDK's token encoding and decoding."""

from llm_sdk import Small_LLM_Model


def main() -> None:
    """Encode a sample request and decode its token sequence."""
    model = Small_LLM_Model()
    text = "Greet shrek"

    token_ids = model.encode(text)

    print(f"Original text: {text!r}")
    print(f"Encoded value: {token_ids}")
    print(f"Python type: {type(token_ids)}")
    print(f"Shape: {token_ids.shape}")

    # encode() returns a batch containing one token sequence.
    # Select that sequence before passing it to decode().
    sequence = token_ids[0]
    decoded_text = model.decode(sequence)
    print(sequence.dtype)       # torch.int64
    for token in token_ids[0]:
        print(f"Python type of token: {type(token)}")
        decoded_token = model.decode(token)
        print(f"Decoded token: {decoded_token!r}")

    print(f"Selected sequence: {sequence}")
    print(f"Sequence shape: {sequence.shape}")
    print(f"Decoded text: {decoded_text!r}")

    # The SDK expects a Python list of token IDs.
    input_ids: list[int] = sequence.tolist()
    logits = model.get_logits_from_input_ids(input_ids)

    # Rank token IDs by their scores, from highest to lowest.
    top_ids = sorted(
        range(len(logits)),
        key=lambda token_id: logits[token_id],
        reverse=True,
    )[:5]

    print(f"\nNumber of candidate tokens: {len(logits)}")
    print("Top 5 next-token candidates:")

    for token_id in top_ids:
        token_text = model.decode([token_id])
        print(
            f"ID: {token_id:>6} | "
            f"Score: {logits[token_id]:>8.3f} | "
            f"Text: {token_text!r}"
        )


if __name__ == "__main__":
    main()
