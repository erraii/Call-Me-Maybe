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
    for token in token_ids[0]:
        decoded_token = model.decode(token)
        print(f"Decoded token: {decoded_token!r}")

    print(f"Selected sequence: {sequence}")
    print(f"Sequence shape: {sequence.shape}")
    print(f"Decoded text: {decoded_text!r}")


if __name__ == "__main__":
    main()
