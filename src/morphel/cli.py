"""MorpheL unified command-line interface.

This module uses Typer to expose all pipeline steps as CLI subcommands.
All expensive operations are preceded by validation from :mod:`morphel.validation`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
import yaml

from morphel import __version__
from morphel.config import MorpheLConfig
from morphel.constants import NUM_SPECIAL_TOKENS
from morphel.datasets import load_text_from_huggingface
from morphel.logging_utils import configure_root_logger, get_logger
from morphel.native_tokenizer import train_morphel
from morphel.reproducibility import collect_run_metadata, save_run_metadata
from morphel.serialization import load_native_tokenizer
from morphel.validation import validate_train_tokenizer_args

app = typer.Typer(
    help="MorpheL: Morphology-Aware Tokenizer Adaptation for Pretrained Models.",
    no_args_is_help=True,
    add_completion=False,
)

logger = get_logger(__name__)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"MorpheL version: {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-v",
            callback=version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = None,
) -> None:
    pass


def load_yaml_config(config_path: Optional[Path]) -> dict[str, Any]:
    if not config_path:
        return {}
    if not config_path.exists():
        typer.secho(f"Error: Config file not found: {config_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    with config_path.open("r", encoding="utf-8") as fh:
        try:
            return yaml.safe_load(fh) or {}
        except yaml.YAMLError as e:
            typer.secho(f"Error parsing YAML config: {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)


@app.command()
def train_tokenizer(
    dataset: Annotated[str, typer.Option(help="Hugging Face dataset identifier")] = "facebook/xnli",
    subset: Annotated[
        str, typer.Option(help="Dataset subset or language code, e.g. tr, ru, fr, vi")
    ] = "",
    language: Annotated[
        str, typer.Option(help="Language identifier stored in morphel_config.json")
    ] = "",
    vowels: Annotated[
        str,
        typer.Option(
            help="Complete target-language vowel inventory. "
            "Example for Turkish: aeiouAEIOUıİöüÖÜâîûÂÎÛ"
        ),
    ] = "",
    splits: Annotated[
        list[str],
        typer.Option(
            help="Explicit tokenizer-induction splits. "
            "Use 'train' for strict induction, or 'train' 'validation' 'test' "
            "to reproduce the original XNLI notebook protocol."
        ),
    ] = [],
    eval_splits: Annotated[
        list[str],
        typer.Option(help="Splits used only for intrinsic tokenizer evaluation."),
    ] = ["validation", "test"],
    text_columns: Annotated[
        list[str],
        typer.Option(help="Dataset columns containing text to tokenize."),
    ] = ["premise", "hypothesis"],
    output_dir: Annotated[
        Optional[Path], typer.Option(help="Directory to save the tokenizer artifacts.")
    ] = None,
    vocab_size: Annotated[int, typer.Option(help="Target nominal vocabulary size.")] = 32_000,
    min_frequency: Annotated[
        int, typer.Option(help="Minimum corpus frequency for a span to enter the vocabulary.")
    ] = 2,
    top_k: Annotated[
        int, typer.Option(help="Maximum positive-MI boundary candidates per word.")
    ] = 4,
    temperature: Annotated[
        float, typer.Option(help="Gumbel sampling temperature. Use 1.0 for training.")
    ] = 1.0,
    mi_threshold: Annotated[
        float, typer.Option(help="Minimum MI score for a boundary to be retained.")
    ] = 0.0,
    model_max_length: Annotated[
        int, typer.Option(help="Maximum sequence length for the HF config.")
    ] = 512,
    seed: Annotated[int, typer.Option(help="Random seed for reproducible sampling.")] = 42,
    disable_vowel_boundary_filter: Annotated[
        bool,
        typer.Option(
            help="Disable the language-specific vowel-transition boundary filter. "
            "Use only when justified and report the change as an ablation."
        ),
    ] = False,
    tokenizer_function: Annotated[
        Optional[str],
        typer.Option(
            help="Import path to a custom text splitting function. "
            "Required for languages without whitespace word boundaries."
        ),
    ] = None,
    hub_repo: Annotated[
        Optional[str],
        typer.Option(help="If provided, push the saved tokenizer to this HF repository ID."),
    ] = None,
    hf_token_env: Annotated[
        str,
        typer.Option(help="Name of the environment variable containing the HF token."),
    ] = "HF_TOKEN",
    config: Annotated[
        Optional[Path], typer.Option(help="Path to a YAML configuration file.")
    ] = None,
    overwrite: Annotated[
        bool, typer.Option(help="Overwrite the output directory if it exists.")
    ] = False,
    num_workers: Annotated[
        int, typer.Option(help="Number of workers for dataset preprocessing.")
    ] = 1,
    cache_dir: Annotated[
        Optional[Path], typer.Option(help="Hugging Face datasets cache directory.")
    ] = None,
    dataset_revision: Annotated[
        Optional[str], typer.Option(help="Dataset revision/commit hash to pin.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option(help="Validate configuration and exit without training.")
    ] = False,
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    """Train a language-configurable MorpheL tokenizer."""
    configure_root_logger(level=log_level)

    # 1. Load config and merge CLI overrides
    yaml_data = load_yaml_config(config)

    def resolve(cli_val: Any, yaml_path: tuple[str, ...], default: Any) -> Any:
        # CLI value wins if it differs from the CLI default, or if there's no YAML value
        # This is a simplified resolution for the script. A robust implementation
        # would inspect whether the CLI argument was explicitly provided.
        # For this setup, we prioritize YAML if the CLI value matches its default.
        val = yaml_data
        for key in yaml_path:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                return cli_val if cli_val else default
        # If CLI provides something truthy and different from default, it wins
        if cli_val and cli_val != default:
            return cli_val
        return val

    resolved_language = resolve(language, ("language", "code"), language)
    resolved_vowels = resolve(vowels, ("language", "vowels"), vowels)
    use_vowel_boundary_filter = not disable_vowel_boundary_filter
    if "use_vowel_boundary_filter" in yaml_data.get("tokenizer", {}):
        use_vowel_boundary_filter = yaml_data["tokenizer"]["use_vowel_boundary_filter"]
    resolved_vocab_size = resolve(vocab_size, ("tokenizer", "vocab_size"), vocab_size)
    resolved_min_freq = resolve(min_frequency, ("tokenizer", "min_frequency"), min_frequency)
    resolved_top_k = resolve(top_k, ("tokenizer", "top_k"), top_k)
    resolved_temp = resolve(temperature, ("tokenizer", "temperature"), temperature)
    resolved_mi = resolve(mi_threshold, ("tokenizer", "mi_threshold"), mi_threshold)
    resolved_max_len = resolve(
        model_max_length, ("tokenizer", "model_max_length"), model_max_length
    )
    resolved_seed = resolve(seed, ("experiment", "seed"), seed)
    resolved_dataset = resolve(dataset, ("dataset", "name"), dataset)
    resolved_subset = resolve(subset, ("dataset", "subset"), subset)
    resolved_splits = resolve(splits, ("dataset", "induction_splits"), splits)
    resolved_eval_splits = resolve(eval_splits, ("dataset", "evaluation_splits"), eval_splits)
    resolved_text_cols = resolve(text_columns, ("dataset", "text_columns"), text_columns)
    resolved_revision = resolve(dataset_revision, ("dataset", "revision"), dataset_revision)

    if not output_dir and "output" in yaml_data and "directory" in yaml_data["output"]:
        output_dir = Path(yaml_data["output"]["directory"])

    if not output_dir:
        typer.secho("Error: --output-dir is required.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # 2. Pre-flight validation
    try:
        validate_train_tokenizer_args(
            language=resolved_language,
            vowels=resolved_vowels,
            use_vowel_boundary_filter=use_vowel_boundary_filter,
            vocab_size=resolved_vocab_size,
            min_frequency=resolved_min_freq,
            top_k=resolved_top_k,
            temperature=resolved_temp,
            output_dir=output_dir,
            overwrite=overwrite,
            splits=resolved_splits,
            num_special_tokens=NUM_SPECIAL_TOKENS,
        )
    except Exception as e:
        typer.secho(f"Validation Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if dry_run:
        typer.echo("Dry run successful. Configuration is valid.")
        raise typer.Exit(code=0)

    # 3. Execution
    morphel_config = MorpheLConfig(
        language=resolved_language,
        vowels=resolved_vowels,
        vocab_size=resolved_vocab_size,
        min_frequency=resolved_min_freq,
        top_k=resolved_top_k,
        temperature=resolved_temp,
        mi_threshold=resolved_mi,
        model_max_length=resolved_max_len,
        seed=resolved_seed,
        use_vowel_boundary_filter=use_vowel_boundary_filter,
    )

    from morphel.preprocessing import simple_tokenize

    tokenizer_fn = simple_tokenize
    if tokenizer_function:
        import importlib

        mod_name, func_name = tokenizer_function.rsplit(":", 1)
        mod = importlib.import_module(mod_name)
        tokenizer_fn = getattr(mod, func_name)

    logger.info("Loading tokenizer-induction corpus...")
    corpus_sentences = load_text_from_huggingface(
        dataset_name=resolved_dataset,
        subset=resolved_subset,
        splits=resolved_splits,
        text_columns=resolved_text_cols,
        revision=resolved_revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        num_workers=num_workers,
    )

    evaluation_sentences = None
    if resolved_eval_splits:
        logger.info("Loading intrinsic-evaluation corpus...")
        evaluation_sentences = load_text_from_huggingface(
            dataset_name=resolved_dataset,
            subset=resolved_subset,
            splits=resolved_eval_splits,
            text_columns=resolved_text_cols,
            revision=resolved_revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            num_workers=num_workers,
        )

    train_morphel(
        corpus_sentences=corpus_sentences,
        config=morphel_config,
        output_dir=output_dir,
        evaluation_sentences=evaluation_sentences,
        tokenizer_fn=tokenizer_fn,
    )

    metadata = collect_run_metadata(
        config_dict=morphel_config.to_dict(),
        seed=resolved_seed,
        dataset_name=resolved_dataset,
        dataset_revision=resolved_revision,
    )
    save_run_metadata(metadata, output_dir)

    if hub_repo:
        import os
        from morphel.hub import push_to_hub

        token = os.environ.get(hf_token_env)
        if not token:
            logger.warning(f"{hf_token_env} is not set; pushing anonymously.")
        push_to_hub(output_dir, hub_repo, token=token)


@app.command()
def inspect_tokenizer(
    tokenizer_dir: Annotated[Path, typer.Option(help="Path to the saved tokenizer directory.")],
) -> None:
    """Inspect a saved MorpheL tokenizer."""
    try:
        artifacts = load_native_tokenizer(tokenizer_dir)
        typer.echo(f"Tokenizer directory: {tokenizer_dir}")
        typer.echo(f"Vocabulary size: {len(artifacts['vocabulary'])}")
        typer.echo(f"Cached word types: {len(artifacts['segmentation_cache'])}")
        typer.echo("Configuration:")
        typer.echo(json.dumps(artifacts["config"].to_dict(), indent=2, ensure_ascii=False))
    except Exception as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command()
def encode(
    tokenizer_dir: Annotated[Path, typer.Option(help="Path to the saved tokenizer directory.")],
    text: Annotated[str, typer.Option(help="Raw text to encode.")],
) -> None:
    """Encode raw text using native MorpheL segmentation."""
    from morphel.native_tokenizer import encode_sentence_as_tokens

    try:
        artifacts = load_native_tokenizer(tokenizer_dir)
        tokens = encode_sentence_as_tokens(
            text,
            vocabulary=artifacts["vocabulary"],
            mi_index=artifacts["mi_index"],
            segmentation_cache=artifacts["segmentation_cache"],
            top_k=artifacts["config"].top_k,
        )
        typer.echo(f"Raw text: {text}")
        typer.echo(f"Tokens:   {tokens}")
    except Exception as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Stubs for other commands (to be fully fleshed out as needed)
# ---------------------------------------------------------------------------

@app.command()
def evaluate_intrinsic() -> None:
    """Compute intrinsic metrics on an existing tokenizer."""
    typer.echo("evaluate-intrinsic not yet implemented in CLI.")

@app.command()
def validate_artifacts() -> None:
    """Verify native artifact integrity."""
    typer.echo("validate-artifacts not yet implemented in CLI.")

@app.command()
def push_to_hub() -> None:
    """Upload tokenizer directory to Hugging Face Hub."""
    typer.echo("push-to-hub not yet implemented in CLI.")

@app.command()
def run_regime_a() -> None:
    """Run Regime A downstream adaptation."""
    typer.echo("run-regime-a not yet implemented in CLI.")

@app.command()
def run_regime_b() -> None:
    """Run Regime B downstream adaptation."""
    typer.echo("run-regime-b not yet implemented in CLI.")

@app.command()
def run_regime_c() -> None:
    """Run Regime C downstream adaptation."""
    typer.echo("run-regime-c not yet implemented in CLI.")

@app.command()
def significance() -> None:
    """Run statistical significance tests."""
    typer.echo("significance not yet implemented in CLI.")

@app.command()
def export_results() -> None:
    """Aggregate results and export tables."""
    typer.echo("export-results not yet implemented in CLI.")


if __name__ == "__main__":
    app()
