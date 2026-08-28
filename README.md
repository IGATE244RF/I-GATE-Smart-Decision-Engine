# I-GATE Reproducible Decision Engine — Release v1.7

This repository accompanies the revised manuscript:

**System Dynamics and Deep Learning Driven Optimization of New Product Development Lifecycle**

The release demonstrates the computational pathway:

NPD data → state representation → LSTM prediction → System Dynamics → optimal-control/policy search → scenario evaluation → gate decision support.

## Scope
The release is a reproducibility demonstration using illustrative data. It is not an industrially calibrated deployment and does not establish empirical reductions in risk, cost, schedule, or resource consumption.

## Main modules
- `src/data_generator.py` — reproducible sequential illustrative data.
- `src/deep_learning.py` — compact LSTM predictor.
- `src/system_dynamics.py` — state-transition model and numerical simulation.
- `src/optimal_control.py` — bounded policy optimization coupled to SD simulation.
- `src/decision_engine.py` — scenario ranking and gate-support logic.
- `examples/run_reproducible_demo.py` — end-to-end demonstration.

## Installation
```bash
pip install -r requirements.txt
```

## Run
```bash
python examples/run_reproducible_demo.py
```

## Reproducibility
The demo uses a fixed random seed. Outputs include an illustrative prediction metric, optimized policy parameter, scenario results, and a trajectory figure.

## Zenodo
After the GitHub release is archived in Zenodo, the permanent DOI should be added here and in `CITATION.cff`. No DOI is claimed before archival publication.
