"""Atomic models for the Lightweaver comparisons."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def crtaf_h6():
    """The CRTAF H_6 used by DexRT (tests/H_6.yaml), loaded into Lightweaver."""
    import pydantic_numpy
    import pydantic_numpy.typing
    # NOTE: pydantic_numpy >= 8 moved the array types; crtaf still imports them
    # from the top level.
    if not hasattr(pydantic_numpy, "NpNDArrayFp64"):
        pydantic_numpy.NpNDArrayFp64 = pydantic_numpy.typing.NpNDArrayFp64
    import crtaf
    from lightweaver.crtaf import from_crtaf
    return from_crtaf(crtaf.Atom.yaml_loads((REPO / "tests/H_6.yaml").read_text()))
