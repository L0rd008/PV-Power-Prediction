# Overleaf Compilation Guide

## Files to upload

| File | Required | Notes |
|------|----------|-------|
| `article.tex` | ✅ | Main article source |
| `article.bib` | ✅ | Bibliography — must be in same folder |

## Overleaf steps

1. Create a new project → **Blank Project**
2. Upload both `article.tex` and `article.bib`
3. Set **Main document** to `article.tex` (Project → Settings)
4. Set **Compiler** to **pdfLaTeX** (recommended) or LuaLaTeX/XeLaTeX
5. Set **Bibliography tool** to **BibTeX**
6. Click **Recompile** — you will need to compile **twice**:
   - First compile: generates `.aux` file
   - `bibtex` run: resolves citations
   - Second compile: embeds bibliography

## Local compilation

```bash
pdflatex article.tex
bibtex article
pdflatex article.tex
pdflatex article.tex
```

## Package dependencies (all standard, on Overleaf by default)

- `palatino`, `microtype` — typography
- `geometry` — page margins
- `xcolor` — colour definitions
- `titlesec` — section heading style
- `parskip` — paragraph spacing
- `enumitem` — list customisation
- `booktabs`, `tabularx`, `array`, `multirow` — tables
- `tcolorbox` (with `skins`, `breakable` libraries) — callout boxes
- `hyperref` — clickable links
- `graphicx`, `float` — figure support
- `fancyhdr` — header/footer
- `biblatex` with `bibtex` backend — bibliography

## Adding figures

To embed a chart (e.g., daily comparison), add to the relevant section:

```latex
\begin{figure}[H]
  \centering
  \includegraphics[width=0.9\linewidth]{daily_comparison.png}
  \caption{Daily actual vs predicted generation — December 2025}
  \label{fig:daily}
\end{figure}
```

Files to consider adding:
- `output/accuracy_comparison.png`
- `output/daily_comparison.png`

## Notes on the article style

- **Document class:** `article` at 11pt A4 — compact, professional
- **Fonts:** Palatino body, bold+color section headings (LinkedIn blue `#0A66C2`)
- **Three callout box types:**
  - `insightbox` — blue, for key insights and justifications
  - `gapbox` — orange, for limitations and warnings
  - `statbox` — green, for headline numbers
- **Citation style:** Numeric (`[1]`), sorted by appearance
- **No abstract:** LinkedIn articles don't use academic abstracts —
  the "About This Article" insightbox serves that role
