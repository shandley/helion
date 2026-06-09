from __future__ import annotations

from functools import cached_property

import numpy as np
import numpy.typing as npt
import torch


class ProteinEmbedder:
    """
    Wraps ESM-2 to produce per-residue embeddings for a protein sequence.

    Uses esm2_t6_8M_UR50D by default -- fast and sufficient for alignment scoring.
    Swap for esm2_t33_650M_UR50D when homology signal at low identity matters more
    than throughput.
    """

    MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
    EMBED_DIM = 320  # matches esm2_t6_8M

    def __init__(self, device: str = "cpu", model_name: str | None = None) -> None:
        self.device = device
        self._model_name = model_name or self.MODEL_NAME

    @cached_property
    def _model_and_tokenizer(
        self,
    ) -> tuple[torch.nn.Module, object]:
        from transformers import EsmModel, EsmTokenizer  # type: ignore[import]

        tokenizer = EsmTokenizer.from_pretrained(self._model_name)
        model = EsmModel.from_pretrained(self._model_name).to(self.device)
        model.eval()
        return model, tokenizer

    def embed(self, sequence: str) -> npt.NDArray[np.float32]:
        """Return per-residue embeddings, shape (n_residues, embed_dim)."""
        model, tokenizer = self._model_and_tokenizer
        inputs = tokenizer(sequence, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = model(**inputs)
        # strip BOS/EOS tokens
        embeddings = output.last_hidden_state[0, 1:-1, :].cpu().numpy()
        return embeddings.astype(np.float32)
