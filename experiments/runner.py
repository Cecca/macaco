import os
import math
import json
import base64
import subprocess
import sys
from datasets import DATASETS
import pprint
import itertools

workers = [
    {"name": name, "port": "2001"}
    for name in [
        "frontend",
        "minion-1",
        "minion-2",
        "minion-3",
        "minion-4",
        "minion-5",
        "minion-6",
        "minion-7",
    ]
]


EXECUTABLE = "target/release/macaco"


def run(configuration):
    with open("/tmp/macaco-current.json", "w") as fp:
        json.dump(configuration, fp)
    conf_str = base64.b64encode(json.dumps(configuration).encode("utf-8"))
    sp = subprocess.run([EXECUTABLE, conf_str])
    if sp.returncode != 0:
        print("Error in invocation with the following configuration")
        print(json.dumps(configuration))
        sys.exit(1)


def run_phones():
    """
    Run experiments on the Wikipedia dataset and its samples
    """
    datasets = [
        "Phones",
        # "Phones-s10000"
    ]
    for dataset in datasets:
        DATASETS[dataset].try_download_preprocessed()
        DATASETS[dataset].preprocess()
    constraints = [
        {
            "stand": 5,
            "null": 5,
            "sit": 5,
            "walk": 5,
            "stairsup": 5,
            "stairsdown": 5,
            "bike": 5,
        }
    ]
    # Fraction of allowed outliers
    frac_outliers = [0.0001]
    fix_outliers = [150, 100, 50]
    fix_outliers = [100]
    # These seeds also define the number of repetitions
    shuffle_seeds = [43234]
    shuffle_seeds = [124351243]
    # shuffle_seeds = [43234, 23562, 12451, 445234, 234524]

    for shuffle_seed, dataset, constr, fix_out in itertools.product(
        shuffle_seeds, datasets, constraints, fix_outliers
    ):
        base_conf = {
            "shuffle_seed": shuffle_seed,
            "outliers": {"Fixed": fix_out},
            # "outliers": {"Percentage": frac_out},
            "dataset": DATASETS[dataset].get_path(),
            "constraint": {"partition": {"categories": constr}},
        }
        # print("Run random")
        # for seed in [1458, 345, 65623]:
        #     c = base_conf.copy()
        #     c["algorithm"] = {"Random": {"seed": seed}}
        #     run(c)

        # if dataset in {"Phones-s10000"}:
        #     # Run the baseline algorithm
        #     c = base_conf.copy()
        #     c["algorithm"] = "ChenEtAl"
        #     run(c)

        for epsilon in [1.0, 0.5, 2.0]:
            c = base_conf.copy()
            c["algorithm"] = {"KaleStreaming": {"epsilon": epsilon}}
            run(c)

        # # Run coreset algorithms
        taus = range(1, 10)
        print(taus)
        for tau in taus:
            # print("Run SeqCoreset", tau)
            c = base_conf.copy()
            c["algorithm"] = {"SeqCoreset": {"tau": tau}}
            # run(c)
            print("Run StreamCoreset", tau)
            c["algorithm"] = {"StreamingCoreset": {"tau": tau}}
            # run(c)

            for hosts in [workers[:i] for i in [1, 2, 4, 8]]:
                print("Run MRCoreset", tau, hosts)
                c = base_conf.copy()
                c["algorithm"] = {"MapReduceCoreset": {"tau": tau}}
                c["parallel"] = {"threads": 1, "hosts": hosts}
                run(c)


def run_higgs():
    """
    Run experiments on the Wikipedia dataset and its samples
    """
    datasets = [
        "Higgs",
        # "Higgs-s10000"
    ]
    for dataset in datasets:
        DATASETS[dataset].try_download_preprocessed()
        DATASETS[dataset].preprocess()
    constraints = [{"signal": 10, "background": 10}]
    # Fraction of allowed outliers
    frac_outliers = [0.0001]
    fix_outliers = [150, 100, 50]
    fix_outliers = [100]
    # These seeds also define the number of repetitions
    shuffle_seeds = [43234]
    shuffle_seeds = [124351243]
    # shuffle_seeds = [43234, 23562, 12451, 445234, 234524]

    for shuffle_seed, dataset, constr, fix_out in itertools.product(
        shuffle_seeds, datasets, constraints, fix_outliers
    ):
        base_conf = {
            "shuffle_seed": shuffle_seed,
            "outliers": {"Fixed": fix_out},
            # "outliers": {"Percentage": frac_out},
            "dataset": DATASETS[dataset].get_path(),
            "constraint": {"partition": {"categories": constr}},
        }
        # print("Run random")
        # for seed in [1458, 345, 65623]:
        #     c = base_conf.copy()
        #     c["algorithm"] = {"Random": {"seed": seed}}
        #     run(c)

        # if dataset in {"Higgs-s10000"}:
        #     # Run the baseline algorithm
        #     c = base_conf.copy()
        #     c["algorithm"] = "ChenEtAl"
        #     run(c)
        for epsilon in [1.0, 0.5, 2.0]:
            c = base_conf.copy()
            c["algorithm"] = {"KaleStreaming": {"epsilon": epsilon}}
            run(c)

        # # Run coreset algorithms
        taus = range(1, 10)
        print(taus)
        for tau in taus:
            # print("Run SeqCoreset", tau)
            c = base_conf.copy()
            c["algorithm"] = {"SeqCoreset": {"tau": tau}}
            # run(c)
            print("Run StreamCoreset", tau)
            c["algorithm"] = {"StreamingCoreset": {"tau": tau}}
            # run(c)

            if dataset not in {"Higgs-s10000"}:
                for hosts in [workers[:i] for i in [1, 2, 4, 8]]:
                    print("Run MRCoreset", tau, hosts)
                    c = base_conf.copy()
                    c["algorithm"] = {"MapReduceCoreset": {"tau": tau}}
                    c["parallel"] = {"threads": 1, "hosts": hosts}
                    run(c)


def run_wiki():
    """
    Run experiments on the Wikipedia dataset and its samples
    """
    datasets = [
        "wiki-d10-c50",
        # "wiki-d10-c50-s10000"
    ]
    for dataset in datasets:
        DATASETS[dataset].try_download_preprocessed()
        DATASETS[dataset].preprocess()
    constraints = [
        # The original matroid constraint, using all the categories
        list(range(0, 50)),
        # Very constrained solution
        # list(range(0, 10)),
    ]
    # Fraction of allowed outliers
    frac_outliers = [0.00001]
    fix_outliers = [150, 100, 50]
    # These seeds also define the number of repetitions
    shuffle_seeds = [43234]
    shuffle_seeds = [124351243]
    # shuffle_seeds = [43234, 23562, 12451, 445234, 234524]

    for shuffle_seed, dataset, constr, fix_out in itertools.product(
        shuffle_seeds, datasets, constraints, fix_outliers
    ):
        base_conf = {
            "shuffle_seed": shuffle_seed,
            "outliers": {"Fixed": fix_out},
            # "outliers": {"Percentage": frac_out},
            "dataset": DATASETS[dataset].get_path(),
            "constraint": {"transversal": {"topics": constr}},
        }
        # Run the naive baseline
        print("Run random")
        # for seed in [1458, 345, 65623]:
        #     c = base_conf.copy()
        #     c["algorithm"] = {"Random": {"seed": seed}}
        #     run(c)

        # if dataset in {
        #     "wiki-d10-c50-s10000"
        # }:
        #     # Run the baseline algorithm
        #     c = base_conf.copy()
        #     c["algorithm"] = "ChenEtAl"
        #     run(c)
        for epsilon in [1.0, 0.5, 2.0]:
            c = base_conf.copy()
            c["algorithm"] = {"KaleStreaming": {"epsilon": epsilon}}
            run(c)

        # # Run coreset algorithms
        taus = range(1, 10)
        print(taus)
        for tau in taus:
            print("Run SeqCoreset", tau)
            c = base_conf.copy()
            c["algorithm"] = {"SeqCoreset": {"tau": tau}}
            # run(c)
            print("Run StreamingCoreset", tau)
            c = base_conf.copy()
            c["algorithm"] = {"StreamingCoreset": {"tau": tau}}
            run(c)
            for hosts in [workers[:i] for i in [1, 2, 4, 8]]:
                print("Run MRCoreset", tau, hosts)
                c = base_conf.copy()
                c["parallel"] = {"threads": 1, "hosts": hosts}
                c["algorithm"] = {"MapReduceCoreset": {"tau": tau}}
                run(c)


if __name__ == "__main__":
    subprocess.run(["cargo", "build", "--release"])
    run_wiki()
    run_higgs()
    run_phones()
