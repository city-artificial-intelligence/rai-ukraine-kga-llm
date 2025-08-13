from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

metrics_dir = Path().resolve() / "metrics"

metric = "F1-Score"
p_val_threshold = 0.05
alternatives = ["greater", "less", "two-sided"]
tests = ["t_test", "wilcoxon"]
float_precision = 4

metrics_df = pd.read_csv(metrics_dir / "OASystems_metrics.csv")

metrics_df = metrics_df[metrics_df["Metric"] == metric]
metric_scores = metrics_df.groupby("System")["Score"].apply(list).to_dict()

systems_with_models = [x for x in metric_scores if "+" in x]
comparators = [k for k in metric_scores if k not in systems_with_models]

columns = pd.MultiIndex.from_product([alternatives, tests])
rows = pd.MultiIndex.from_product([systems_with_models, comparators])

pval_df = pd.DataFrame(index=rows, columns=columns)
stat_df = pd.DataFrame(index=rows, columns=columns)
decision_df = pd.DataFrame(index=rows, columns=columns)

for model in systems_with_models:
    ours = metric_scores[model]
    for comparator in comparators:
        others = metric_scores[comparator]
        for alt in alternatives:
            for test_name, test_fn in [("t_test", ttest_rel), ("wilcoxon", wilcoxon)]:
                stat, pval = test_fn(ours, others, alternative=alt)
                pval_df.loc[(model, comparator), (alt, test_name)] = round(pval, float_precision)
                stat_df.loc[(model, comparator), (alt, test_name)] = round(stat, float_precision)
                decision_df.loc[(model, comparator), (alt, test_name)] = (
                    "significant" if pval < p_val_threshold else "n.s."
                )

for df, name in zip([pval_df, stat_df, decision_df], ["p_values", "test_statistics", "decisions"]):
    save_path = metrics_dir / "statistical-testing" / f"Systems-Analysis-{name}.csv"
    df.to_csv(save_path, sep="\t")
    print(f"Saved {name} to {save_path}")
