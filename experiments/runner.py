import math
import json
import base64
import subprocess
import sys
from datasets import DATASETS
import pprint
import itertools


EXECUTABLE = "target/release/kcmkc"


def run(configuration):
    conf_str = base64.b64encode(json.dumps(configuration).encode("utf-8"))
    sp = subprocess.run([EXECUTABLE, conf_str])
    if sp.returncode != 0:
        print("Error in invocation with the following configuration")
        print(json.dumps(configuration))
        sys.exit(1)


def run_wiki():
    """
    Run experiments on the Wikipedia dataset and its samples
    """
    datasets = [
        "wiki-d50-c100-s50000",  # <- a sample where we can also run the baseline algorithm
        "wiki-d50-c100-s50000-eucl",  # <- a sample where we can also run the baseline algorithm, euclidean distance
        "wiki-d50-c100",  # <- The full wikipedia dataset
        "wiki-d50-c100-eucl",  # <- The full wikipedia dataset, euclidean distance
    ]
    for dataset in datasets:
        DATASETS[dataset].preprocess()
    constraints = [
        # Very constrained solution
        list(range(0, 10)),
        # The original matroid constraint, using all the categories
        # list(range(0, 100)),
    ]
    # Fraction of allowed outliers
    frac_outliers = [0.01]
    # These seeds also define the number of repetitions
    shuffle_seeds = [43234, 23562, 12451, 445234, 234524]

    for dataset, constr, frac_out, shuffle_seed in itertools.product(
        datasets, constraints, frac_outliers, shuffle_seeds
    ):
        # Run the naive baseline
        for seed in [1458, 345, 65623]:
            run(
                {
                    "shuffle_seed": shuffle_seed,
                    "outliers": {"Percentage": frac_out},
                    "algorithm": {"Random": {"seed": seed}},
                    "dataset": DATASETS[dataset].get_path(),
                    "constraint": {"transversal": {"topics": constr}},
                }
            )

        if dataset in {"wiki-d50-c100-s50000"}:
            # Run the baseline algorithm
            run(
                {
                    "shuffle_seed": shuffle_seed,
                    "outliers": {"Percentage": frac_out},
                    "algorithm": "ChenEtAl",
                    "dataset": DATASETS[dataset].get_path(),
                    "constraint": {"transversal": {"topics": constr}},
                }
            )

        # Run coreset algorithms
        taus = [2 ** x for x in [9, 10, 11, 12]]
        print(taus)
        for tau in taus:
            run(
                {
                    "shuffle_seed": shuffle_seed,
                    "outliers": {"Percentage": frac_out},
                    "algorithm": {"SeqCoreset": {"tau": tau}},
                    "dataset": DATASETS[dataset].get_path(),
                    "constraint": {"transversal": {"topics": constr}},
                }
            )
            run(
                {
                    "shuffle_seed": shuffle_seed,
                    "outliers": {"Percentage": frac_out},
                    "algorithm": {"StreamingCoreset": {"tau": tau}},
                    "dataset": DATASETS[dataset].get_path(),
                    "constraint": {"transversal": {"topics": constr}},
                }
            )
            for threads in [2, 4, 8, 16]:
                # Keep the size of the final coreset constant across thread counts
                rescaled_tau = int(math.ceil(tau / threads))
                print("tau", tau, "threads", threads, "rescaled", rescaled_tau)
                run(
                    {
                        "parallel": {"threads": threads},
                        "shuffle_seed": shuffle_seed,
                        "outliers": {"Percentage": frac_out},
                        "algorithm": {"MapReduceCoreset": {"tau": rescaled_tau}},
                        "dataset": DATASETS[dataset].get_path(),
                        "constraint": {"transversal": {"topics": constr}},
                    }
                )


def run_exploration():
    for dataset in ["wiki-d50-c100-s10000"]:
        for frac_outlier in [0.1]:
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
                # for epsilon in [0.1]:
                #     run(
                #         {
                #             "outliers": {"Percentage": frac_outlier},
                #             "algorithm": {"SeqCoreset": {"epsilon": epsilon}},
                #             "dataset": DATASETS[dataset].get_path(),
                #             "constraint": {"transversal": {"topics": topics}},
                #         }
                #     )
                # Streaming coreset
                for size in [100]:
                    run(
                        {
                            "outliers": {"Percentage": frac_outlier},
                            "algorithm": {"StreamingCoreset": {"coreset_size": size}},
                            "dataset": DATASETS[dataset].get_path(),
                            "constraint": {"transversal": {"topics": topics}},
                        }
                    )


if __name__ == "__main__":
    subprocess.run(["cargo", "build", "--release"])
    run_wiki()
