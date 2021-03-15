import json
import base64
import subprocess
import sys
from datasets import DATASETS


EXECUTABLE = "target/release/kcmkc"


def run(configuration):
    conf_str = base64.b64encode(json.dumps(configuration).encode("utf-8"))
    subprocess.run([EXECUTABLE, conf_str])


def run_exploration():
    for dataset in ["wiki-d50-c100-s10000"]:
        for frac_outlier in [0.01, 0.1]:
            for topics in [list(range(20)), list(range(50)), list(range(100))]:
                # Run the random algorithm, which sets the baseline
                for seed in [2509867293, 356235, 256, 23561, 14646, 14656, 3562456]:
                    run(
                        {
                            "outliers": {"Percentage": frac_outlier},
                            "algorithm": {"Random": {"seed": seed}},
                            "dataset": DATASETS[dataset].get_path(),
                            "constraint": {"transversal": {"topics": topics}},
                        }
                    )
                # Run Chen et al
                run(
                    {
                        "outliers": {"Percentage": frac_outlier},
                        "algorithm": "ChenEtAl",
                        "dataset": DATASETS[dataset].get_path(),
                        "constraint": {"transversal": {"topics": topics}},
                    }
                )
                # Sequential coreset
                for epsilon in [0.1, 0.5, 1, 1.5]:
                    run(
                        {
                            "outliers": {"Percentage": frac_outlier},
                            "algorithm": {"SeqCoreset": {"epsilon": epsilon}},
                            "dataset": DATASETS[dataset].get_path(),
                            "constraint": {"transversal": {"topics": topics}},
                        }
                    )


if __name__ == "__main__":
    subprocess.run(["cargo", "build", "--release"])
    run_exploration()
