"""Classical baselines.

Ownership: Dan.

The inputs are audio clips, so a text TF-IDF pipeline is not a baseline
for this dataset. The audio representation (for example MFCCs or another
acoustic summary) and the classifiers have not been chosen. Nothing in this
module is implemented.

When baselines are added, they must use the shared split in
``data/splits/shared_split.csv``. They must not draw a new split. The
internal test split is only for final comparison.
"""
