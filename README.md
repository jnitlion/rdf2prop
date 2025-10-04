<<<<<<< HEAD
# rdf2prop
Predict viscosity and conductivity from structure (RDFs)
=======
# rdf2prop — Predict viscosity and conductivity from structure (RDFs)

**rdf2prop** converts MD-derived structure into features and trains supervised models to predict:
- **Viscosity** η (Pa·s) and/or
- **Conductivity** σ (S/m)

The model uses **radial distribution functions (RDFs)**, pair-correlation entropy (S₂), a coordination-number proxy, and metadata. Training is done in **log-space** (`ln(η)`, `ln(σ)`) for stability; we also report **real-unit** errors.

> ✅ This workflow requires **one full RDF per state point**, with columns `r,g`.  

---

## 1) Requirements

- Python ≥ 3.9  
- Packages: `numpy, pandas, scipy, scikit-learn, xgboost, matplotlib, shap`

Install (recommended: virtual environment):

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

>>>>>>> 2232e89 (feat: initial public release (full RDF only) with README, requirements, and structure)
