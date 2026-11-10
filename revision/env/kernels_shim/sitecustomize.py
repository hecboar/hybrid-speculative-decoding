# sitecustomize.py — auto-imported by Python at interpreter startup when this
# directory is on PYTHONPATH. It aliases generation-output symbols that OLD
# mamba-ssm imports (e.g. GreedySearchDecoderOnlyOutput) to their renamed modern
# equivalents in transformers.generation, so `import mamba_ssm` succeeds against
# a modern (pinned) transformers WITHOUT modifying either library on disk.
# Applied only as Step 2 of the E_KERNELS escalation ladder. No-op if the modern
# names are absent or the old names already exist.
try:
    import transformers.generation as _g

    _ALIASES = {
        "GreedySearchDecoderOnlyOutput": "GenerateDecoderOnlyOutput",
        "GreedySearchEncoderDecoderOutput": "GenerateEncoderDecoderOutput",
        "SampleDecoderOnlyOutput": "GenerateDecoderOnlyOutput",
        "SampleEncoderDecoderOutput": "GenerateEncoderDecoderOutput",
        "BeamSearchDecoderOnlyOutput": "GenerateBeamDecoderOnlyOutput",
        "BeamSearchEncoderDecoderOutput": "GenerateBeamEncoderDecoderOutput",
        "BeamSampleDecoderOnlyOutput": "GenerateBeamDecoderOnlyOutput",
        "BeamSampleEncoderDecoderOutput": "GenerateBeamEncoderDecoderOutput",
        "ContrastiveSearchDecoderOnlyOutput": "GenerateDecoderOnlyOutput",
        "ContrastiveSearchEncoderDecoderOutput": "GenerateEncoderDecoderOutput",
    }
    for _old, _new in _ALIASES.items():
        if not hasattr(_g, _old) and hasattr(_g, _new):
            setattr(_g, _old, getattr(_g, _new))
except Exception:
    # Never let the shim break interpreter startup for unrelated processes.
    pass
