# Repository scope and curation policy

## Retained

- The complete `src/clarifysae_llama/` package, because the runners share
  backends, evaluation code, ClarQ interaction code, SAE utilities, and steering
  implementations.
- Small prompt and vocabulary assets needed by CLAM and ClarifyScore.
- Representative configurations for all major future experiment families.
- Helper scripts that prepare discovery data, build decoder-similarity feature
  sets, create and validate CLAM demonstrations, and launch runs/sweeps.
- The Anthropic-inspired additions and their tests.

## Omitted

- `clam/`: historical predictions, metrics, and tables. These are experiment
  outputs, not executable inputs.
- `visualization/data/`: historical result bundles and archives.
- `visualization/*.py` and visualization-only YAML files: not required to run or
  evaluate the proposed experiments; the maintained evaluation package already
  emits CSV/JSON/HTML reports where applicable.
- `src/*.egg-info/`: generated packaging metadata recreated by installation.
- `notebooks/`: contained no experiment code.
- `clam_zero_shot_stage2_final.patch`: an old patch, superseded by the current
  source tree.
- Redundant or obsolete YAML variants: representative templates are retained,
  and new layers/features/strengths should be expressed by editing or copying
  those templates.
- The older duplicate LMSYS preparation script and result-compaction script.

## Size and reproducibility trade-off

The source package is retained in full even though some modules will not be used
in every experiment. It is small and avoids silently breaking shared imports or
removing evaluation paths needed for AmbiK, ClarQ-LLM, or CLAM. Large generated
artifacts are excluded because they can be regenerated and do not belong in a
clean experimental source repository.
