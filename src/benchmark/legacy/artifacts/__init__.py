"""Generate LaTeX tables and figures for research papers."""

from .latex import generate_latex_tables, format_cell
from .figures import generate_figures, plot_comparison, plot_convergence

__all__ = [
    "generate_latex_tables",
    "format_cell",
    "generate_figures",
    "plot_comparison",
    "plot_convergence",
]
