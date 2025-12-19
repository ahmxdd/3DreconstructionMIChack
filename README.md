Here is a concise `README.md` ready for your hackathon submission.

```markdown
# Open-Source 3-D Membrane Reconstruction from a Single 2-D FIB-SEM Micrograph

A computational framework for reconstructing statistically accurate 3D porous structures from single 2D images using Gaussian Random Fields and validating them via Pore Network Modeling.

### 1. Interactive 3D Demo
Create a virtual environment of your choosing, run `pip install numpy matplotlib scipy scikit-image porespy openpnm streamlit plotly`, and run:
```bash
streamlit run demo.py

```

### 2. Core Reconstruction (Tasks 1-3)

To run the full parameter optimization and generation pipeline:

```bash
python task3.py

```

### 3. Physical Evaluations

To perform physics simulations (permeability, tortuosity) and generate validation plots:

```bash
python evals.py

```

## 📚 References

**[1] Code Implementation:**
H. Chamani, A. Rabbani, K. P. Russell, A. L. Zydney, E. D. Gomez, J. Hattrick-Simpers, and J. R. Werber, “Rapid reconstruction of 3-D membrane pore structure using a single 2-D micrograph,” *arXiv preprint arXiv:2301.10601*, 2023. Available: https://arxiv.org/abs/2301.10601

**[2] Dataset:**
K. P. Brickey, A. L. Zydney, and E. D. Gomez, “FIB-SEM tomography reveals the nanoscale 3D morphology of virus removal filters,” *Journal of Membrane Science*, vol. 640, p. 119766, 2021. Available: https://par.nsf.gov/servlets/purl/10308587

```

```
