# ML4H manuscript bundle

Place these files in the existing `paper/` directory. Keep the official
`jmlr.cls` and `jmlrutils.sty` files unchanged.

Expected structure:

```text
paper/
├── main.tex
├── references.bib
├── jmlr.cls
├── jmlrutils.sty
└── figures/
    ├── figure1_study_design.pdf
    └── figure2_fixed_budget_capture.pdf
```

The SVG and PNG figure versions are included as editable/archive copies;
`main.tex` uses the PDF versions.

Compile with PDFLaTeX:

```zsh
cd paper
latexmk -C
rm -f main.bbl main.blg
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Check for serious layout or citation problems:

```zsh
grep -Ei "undefined citation|undefined reference|overfull|latex error|package .* error" main.log
```

The supplied preview was compiled and visually inspected using a stock JMLR
class with local ML4H macro shims. The delivered `main.tex` targets the
official ML4H 2026 class already present in the project.

Before submission, confirm:

1. `\mlhtrack{proceedings}` is the intended track.
2. The IRB wording matches the applicable institutional determination.
3. The review repository/code package is anonymized.
