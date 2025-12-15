import matplotlib.pyplot as plt

# Default settings: no LaTeX
_USE_TEX = False
_FONT_FAMILY = "serif"
_MATH_FONTSET = "stix"

def set_latex(
    use_tex: bool = False,
    font_family: str = "serif",
    math_fontset: str = "stix"
) -> None:
    """
    Configure matplotlib to render text with (or without) LaTeX.

    Parameters
    ----------
    use_tex : bool
        Whether to enable full LaTeX rendering (text.usetex).
        Defaults to False (use mathtext).
    font_family : str
        The font.family to apply.
    math_fontset : str
        The mathtext.fontset to use when not using full LaTeX.
    """
    global _USE_TEX, _FONT_FAMILY, _MATH_FONTSET
    _USE_TEX = use_tex
    _FONT_FAMILY = font_family
    _MATH_FONTSET = math_fontset

    try:
        plt.rcParams.update({
            "text.usetex": _USE_TEX,
            "font.family": _FONT_FAMILY
        })
        if not _USE_TEX:
            plt.rcParams["mathtext.fontset"] = _MATH_FONTSET
    except RuntimeError:
        # Fallback if full LaTeX fails
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": _FONT_FAMILY,
            "mathtext.fontset": _MATH_FONTSET
        })


# Apply defaults (no LaTeX) at import time
set_latex(_USE_TEX, _FONT_FAMILY, _MATH_FONTSET)


# Mapping for escaping special LaTeX chars
_ESC = {
    '&':  r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
    '_':  r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
    '^':  r'\textasciicircum{}', '\\': r'\textbackslash{}',
}

def _esc(txt: str) -> str:
    """Escape characters so they render correctly in LaTeX/mathtext."""
    return ''.join(_ESC.get(c, c) for c in str(txt))