"""Qwen3 model loading and decoder-block utilities.

The project deliberately pins a Qwen3-capable Transformers release.  Keeping
all model-internal access in this module makes failures from a future HF API
change early and actionable instead of silently corrupting an experiment.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from torch import nn


def resolve_torch_dtype(name: str) -> torch.dtype | str:
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name.lower() == "auto":
        return "auto"
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        choices = ["auto", *sorted(aliases)]
        raise ValueError(f"Unsupported dtype {name!r}; choose one of {choices}") from exc


def load_causal_lm(
    model_name_or_path: str,
    *,
    revision: str | None = None,
    dtype: str = "bf16",
    device: str | torch.device | None = None,
    local_files_only: bool = False,
    attn_implementation: str | None = None,
):
    """Load a causal LM with conservative, reproducible defaults."""

    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "revision": revision,
        "dtype": resolve_torch_dtype(dtype),
        "local_files_only": local_files_only,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    validate_qwen3(model)
    if device is not None:
        model.to(device)
    return model


def load_tokenizer(
    model_name_or_path: str,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        revision=revision,
        local_files_only=local_files_only,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def validate_qwen3(model: nn.Module) -> None:
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type != "qwen3":
        raise TypeError(f"This implementation requires a Qwen3 causal LM, got {model_type!r}")
    layers = get_layers(model)
    configured = int(getattr(model.config, "num_hidden_layers", -1))
    if configured != len(layers):
        raise ValueError(
            f"Model exposes {len(layers)} decoder layers but config says {configured}"
        )


def get_backbone(model: nn.Module) -> nn.Module:
    backbone = getattr(model, "model", None)
    if backbone is None or not hasattr(backbone, "layers"):
        raise TypeError("Expected a Hugging Face causal LM with model.layers")
    return backbone


def get_layers(model: nn.Module) -> nn.ModuleList:
    layers = get_backbone(model).layers
    if not isinstance(layers, nn.ModuleList):
        raise TypeError(f"Expected model.layers to be ModuleList, got {type(layers).__name__}")
    return layers


def reindex_layers(model: nn.Module) -> None:
    """Update cache indices and config after changing model depth."""

    layers = get_layers(model)
    for index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        if attention is not None and hasattr(attention, "layer_idx"):
            attention.layer_idx = index
    model.config.num_hidden_layers = len(layers)
    backbone = get_backbone(model)
    if hasattr(backbone, "config"):
        backbone.config.num_hidden_layers = len(layers)
    layer_types = getattr(model.config, "layer_types", None)
    if isinstance(layer_types, list) and len(layer_types) != len(layers):
        model.config.layer_types = [getattr(layer, "attention_type", "full_attention") for layer in layers]


def set_layers(model: nn.Module, layers: Sequence[nn.Module]) -> None:
    get_backbone(model).layers = nn.ModuleList(list(layers))
    reindex_layers(model)


def clone_block(block: nn.Module) -> nn.Module:
    return copy.deepcopy(block)


def freeze(module: nn.Module) -> nn.Module:
    module.requires_grad_(False)
    module.eval()
    return module


def train_only(module: nn.Module, selected: Iterable[nn.Module]) -> None:
    module.requires_grad_(False)
    for item in selected:
        item.requires_grad_(True)


def block_forward(
    model: nn.Module,
    block: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Run one Qwen3 decoder block on a cached boundary state.

    The causal mask and rotary embeddings must match the full-model forward.
    This implementation targets the pinned Transformers API and raises a clear
    error if that API moves.
    """

    try:
        from transformers.masking_utils import create_causal_mask
    except ImportError:  # pragma: no cover - protects future dependency changes
        try:
            from transformers.models.qwen3.modeling_qwen3 import create_causal_mask
        except ImportError as exc:
            raise RuntimeError(
                "Could not import Transformers' causal-mask helper; install the pinned requirements"
            ) from exc

    backbone = get_backbone(model)
    batch, sequence, _ = hidden_states.shape
    device = hidden_states.device
    cache_position = torch.arange(sequence, device=device)
    position_ids = cache_position.unsqueeze(0)
    position_embeddings = backbone.rotary_emb(hidden_states, position_ids)
    causal_mask = create_causal_mask(
        config=backbone.config,
        input_embeds=hidden_states,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids,
    )
    output = block(
        hidden_states,
        attention_mask=causal_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
    )
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Decoder block returned {type(output).__name__}, expected Tensor")
    if output.shape != (batch, sequence, hidden_states.shape[-1]):
        raise ValueError(f"Unexpected block output shape {tuple(output.shape)}")
    return output


@contextmanager
def capture_layer_inputs(model: nn.Module) -> Iterator[list[torch.Tensor]]:
    """Capture exact input states at every decoder boundary during a forward.

    Transformers' public ``hidden_states`` tuple normalizes the final element;
    forward pre-hooks instead capture the raw inputs to each block.  Because the
    final block is protected, these inputs plus non-final block outputs provide
    every boundary required by the experiment without ambiguity.
    """

    captured: list[torch.Tensor] = []
    handles = []

    def hook(_module, args, kwargs):
        hidden = kwargs.get("hidden_states") if kwargs else None
        if hidden is None and args:
            hidden = args[0]
        if hidden is None:
            raise RuntimeError("Could not capture decoder-layer hidden_states")
        captured.append(hidden.detach())

    for layer in get_layers(model):
        handles.append(layer.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def teacher_boundaries(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    """Return every raw block boundary ``h_0..h_L`` (before final norm)."""

    layers = get_layers(model)
    final_outputs: list[torch.Tensor] = []

    def capture_final(_module, _args, output):
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("Could not capture final decoder-layer output")
        final_outputs.append(value.detach())

    handle = layers[-1].register_forward_hook(capture_final)
    try:
        # Calling the backbone avoids materializing the very large vocabulary
        # logits.  The layer hooks still see exactly the same raw block inputs
        # and final block output as a full causal-LM forward.
        with torch.inference_mode(), capture_layer_inputs(model) as captured:
            get_backbone(model)(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
    finally:
        handle.remove()
    expected = len(layers)
    if len(captured) != expected or len(final_outputs) != 1:
        raise RuntimeError(
            f"Captured {len(captured)} inputs and {len(final_outputs)} final outputs; "
            f"expected {expected} and 1"
        )
    return (*captured, final_outputs[0])


def state_dict_to_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def load_block_state(block: nn.Module, path_or_state: str | Path | Mapping[str, torch.Tensor]) -> None:
    if isinstance(path_or_state, (str, Path)):
        state = torch.load(path_or_state, map_location="cpu", weights_only=True)
        if isinstance(state, Mapping) and "model" in state:
            state = state["model"]
    else:
        state = path_or_state
    block.load_state_dict(state, strict=True)
