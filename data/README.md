# Dataset folder (not committed to GitHub)

Place the Kaggle CloudTrail dataset files here:

- `dec12_18features.csv`       (~793 MB)
- `nineteenFeaturesDf.csv`     (~994 MB, this is the one the pipeline uses)

Source: Kaggle — *"AWS CloudTrail Dataset from flaws.cloud"*
(`archive.zip` contains both CSVs; extract it into this folder).

Expected layout:

```
data/
├── README.md                (this file)
├── dec12_18features.csv     ← you add
└── nineteenFeaturesDf.csv   ← you add
```

CSV files are gitignored (GitHub 100 MB limit); everything else needed to run
the pipeline (code, trained model, thresholds, MITRE config) is in the repo.
