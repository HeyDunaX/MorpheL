"""MorpheL downstream adaptation regimes.

The paper defines three adaptation regimes (A, B, C) that control which
parameters are trainable and frozen when integrating a new tokenizer into a
pretrained encoder (e.g., XLM-R).

- **Regime A**: Only the classifier head is trainable. Encoder and word
  embeddings are frozen. Requires exact mapping initialization (e.g., WECHSEL).
- **Regime B**: Classifier head and word embeddings are trainable. Encoder body
  is frozen. Gradients must flow back to the embeddings.
- **Regime C**: Full fine-tuning. Classifier head, word embeddings, and the
  entire encoder body are trainable.

Important
---------
The AGENT_HANDOFF.md specifies that the official canonical implementations for
Regime A, B, and C are NOT included in the handoff bundle. This package
provides the documented interface and stubs. The project owner will replace
the stubs with the official PyTorch code before running experiments.
"""
