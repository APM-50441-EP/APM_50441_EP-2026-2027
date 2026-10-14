# APM_50441_EP - Machine Learning 2 - Labs (2026-2027)

The labs of the course, one folder per session (`PC1` to `PC7`): the lab notebooks, the lab handout (`LabN.pdf`), the
exercise sheet (`PCn.pdf`) and the data. The solutions are added after each session.

## Working environment

- **Colab** (the simplest option): open a notebook from its *Open in Colab* badge, or at
  `https://colab.research.google.com/github/APM-50441-EP/APM_50441_EP-2026-2027/blob/main/PCn/<notebook>.ipynb`. The R labs (2, 4 and 7)
  need the R runtime (*Runtime > Change runtime type*); their first cell installs the packages.
- **On your own computer**: clone this repository (or download a lab folder) and open the notebooks in an IDE that
  runs Jupyter notebooks (VS Code, JupyterLab, Spyder, PyCharm), or the `.qmd` files of the R labs in RStudio or
  Positron. The Python environment is described in `env/pyproject.toml` (`uv sync` in `env`), the R packages are
  installed with `env/install_packages.R` (`install_lab("lab2")`, ...). The data are read from the local files when
  they are present, otherwise downloaded.

Each lab has a core part (about 2 hours) and independent *To go further* blocks; `*_simplified.ipynb` is the core
part only.

## Use of AI and license

Artificial intelligence tools have been used to revise the content of these labs. The main contributor takes full
responsibility for this content.

The texts are under the Creative Commons Attribution-ShareAlike 4.0 license
([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)), and the code under the MIT license (see `LICENSE`);
the datasets keep their own licenses. Main contributor: E. Le Pennec. Contributors: S. Boucheron, A. Dieuleveut,
A.K. Fermin, S. Gadat, S. Gaiffas, B. Goujaud, A. Guilloux, Ch. Keribin, Th. Klein, E. Matzner, J. Read, M. Sangnier,
E. Scornet.
