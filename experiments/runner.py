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
    with open("/tmp/kcmkc-current.json", "w") as fp:
        json.dump(configuration, fp)
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
        "wiki-d50-c100-s10000",  # <- a sample where we can also run the baseline algorithm
        # "wiki-d50-c100-s10000-eucl",  # <- a sample where we can also run the baseline algorithm, euclidean distance
        "wiki-d50-c100",  # <- The full wikipedia dataset
        # "wiki-d50-c100-eucl",  # <- The full wikipedia dataset, euclidean distance
    ]
    for dataset in datasets:
        DATASETS[dataset].try_download_preprocessed()
        DATASETS[dataset].preprocess()
    constraints = [
        # The original matroid constraint, using all the categories
        list(range(0, 100)),
        # Very constrained solution
        list(range(0, 10)),
    ]
    # Fraction of allowed outliers
    frac_outliers = [0.01]
    # These seeds also define the number of repetitions
    shuffle_seeds = [43234, 23562, 12451, 445234, 234524]

    for shuffle_seed, dataset, constr, frac_out in itertools.product(
        shuffle_seeds, datasets, constraints, frac_outliers
    ):
        # Run the naive baseline
        print("Run random")
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

        # if dataset in {"wiki-d50-c100-s10000", "wiki-d50-c100-s10000-eucl"}:
            # Run the baseline algorithm
            # run(
            #     {
            #         "shuffle_seed": shuffle_seed,
            #         "outliers": {"Percentage": frac_out},
            #         "algorithm": "ChenEtAl",
            #         "dataset": DATASETS[dataset].get_path(),
            #         "constraint": {"transversal": {"topics": constr}},
            #     }
            # )

        # # Run coreset algorithms
        taus = [2 ** x for x in [3, 4, 5, 6]]
        print(taus)
        for tau in taus:
            print("Run SeqCoreset", tau)
            run(
                {
                    "shuffle_seed": shuffle_seed,
                    "outliers": {"Percentage": frac_out},
                    "algorithm": {"SeqCoreset": {"tau": tau}},
                    "dataset": DATASETS[dataset].get_path(),
                    "constraint": {"transversal": {"topics": constr}},
                }
            )
            print("Run StreamCoreset", tau)
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
                print("Run MRCoreset", tau, threads)
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


def check():
    datasets = [
        # "wiki-d50-c100-s10000",  # <- a sample where we can also run the baseline algorithm
        "wiki-d50-c100",  # <- The full wikipedia dataset
    ]
    for dataset in datasets:
        DATASETS[dataset].try_download_preprocessed()
        DATASETS[dataset].preprocess()
    constraints = [
        # The original matroid constraint, using all the categories
        list(range(0, 100)),
        # # Very constrained solution
        # list(range(0, 10)),
    ]
    # Fraction of allowed outliers
    frac_outliers = [0.01]
    # These seeds also define the number of repetitions
    # shuffle_seeds = [43234, 23562, 12451, 445234, 234524]
    shuffle_seeds = [43234]

    for shuffle_seed, dataset, constr, frac_out in itertools.product(
        shuffle_seeds, datasets, constraints, frac_outliers
    ):
        # Run coreset algorithms
        taus = [16, 512]
        print(taus)
        for tau in taus:
            run(
                {
                    "shuffle_seed": shuffle_seed,
                    "outliers": {"Percentage": frac_out},
                    "algorithm": {"StreamingCoreset": {"tau": tau}},
                    "dataset": DATASETS[dataset].get_path(),
                    "constraint": {"transversal": {"topics": constr}},
                }
            )


if __name__ == "__main__":
    subprocess.run(["cargo", "build", "--release"])
    run_wiki()
