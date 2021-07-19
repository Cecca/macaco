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


def run_higgs():
    """
    Run experiments on the Wikipedia dataset and its samples
    """
    datasets = ["Higgs"]
    for dataset in datasets:
        DATASETS[dataset].try_download_preprocessed()
        DATASETS[dataset].preprocess()
    constraints = [{"signal": 10, "background": 10}]
    # Fraction of allowed outliers
    frac_outliers = [0.0001]
    # These seeds also define the number of repetitions
    shuffle_seeds = [43234]
    # shuffle_seeds = [43234, 23562, 12451, 445234, 234524]

    for shuffle_seed, dataset, constr, frac_out in itertools.product(
        shuffle_seeds, datasets, constraints, frac_outliers
    ):
        base_conf = {
            "shuffle_seed": shuffle_seed,
            "outliers": {"Percentage": frac_out},
            "dataset": DATASETS[dataset].get_path(),
            "constraint": {"partition": constr},
        }
        # Run the naive baseline
        print("Run random")
        for seed in [1458, 345, 65623]:
            c = base_conf.copy()
            c["algorithm"] = {"Random": {"seed": seed}}
            run(c)

        # if dataset in {
        # }:
        #     # Run the baseline algorithm
        #     run(
        #         {
        #             "shuffle_seed": shuffle_seed,
        #             "outliers": {"Percentage": frac_out},
        #             "algorithm": "ChenEtAl",
        #             "dataset": DATASETS[dataset].get_path(),
        #             "constraint": constr,
        #         }
        #     )

        # # Run coreset algorithms
        taus = [2 ** x for x in [3, 6]]
        print(taus)
        for tau in taus:
            print("Run SeqCoreset", tau)
            c = base_conf.copy()
            c["algorithm"] = {"SeqCoreset": {"tau": tau}}
            run(c)
            print("Run StreamCoreset", tau)
            c["algorithm"] = {"StreamingCoreset": {"tau": tau}}
            run(c)

            for hosts in [workers[:i] for i in [2, 4, 8]]:
                print("Run MRCoreset", tau, hosts)
                c["algorithm"] = {"MapReduceCoreset": {"tau": tau}}
                c["parallel"] = {"threads": 1, "hosts": hosts}
                run(c)


def run_wiki():
    """
    Run experiments on the Wikipedia dataset and its samples
    """
    datasets = [
        # "wiki-d10-c50",
        "wiki-d10-c50-s10000"
        # "wiki-d50-c100-s10000",  # <- a sample where we can also run the baseline algorithm
        # "wiki-d50-c100-s10000-eucl",  # <- a sample where we can also run the baseline algorithm, euclidean distance
        # "wiki-d50-c100",  # <- The full wikipedia dataset
        # "wiki-d50-c100-eucl",  # <- The full wikipedia dataset, euclidean distance
    ]
    for dataset in datasets:
        DATASETS[dataset].try_download_preprocessed()
        DATASETS[dataset].preprocess()
    constraints = [
        # The original matroid constraint, using all the categories
        # list(range(0, 50)),
        # Very constrained solution
        list(range(0, 10)),
    ]
    # Fraction of allowed outliers
    frac_outliers = [10]
    # These seeds also define the number of repetitions
    shuffle_seeds = [43234]
    # shuffle_seeds = [43234, 23562, 12451, 445234, 234524]

    for shuffle_seed, dataset, constr, frac_out in itertools.product(
        shuffle_seeds, datasets, constraints, frac_outliers
    ):
        # Run the naive baseline
        print("Run random")
        for seed in [1458, 345, 65623]:
            run(
                {
                    "shuffle_seed": shuffle_seed,
                    "outliers": {"Fixed": frac_out},
                    "algorithm": {"Random": {"seed": seed}},
                    "dataset": DATASETS[dataset].get_path(),
                    "constraint": {"transversal": {"topics": constr}},
                }
            )

        if dataset in {
            "wiki-d10-c50-s10000",
            "wiki-d50-c100-s10000",
            "wiki-d50-c100-s10000-eucl",
        }:
            # Run the baseline algorithm
            run(
                {
                    "shuffle_seed": shuffle_seed,
                    "outliers": {"Fixed": frac_out},
                    "algorithm": "ChenEtAl",
                    "dataset": DATASETS[dataset].get_path(),
                    "constraint": {"transversal": {"topics": constr}},
                }
            )

        # # Run coreset algorithms
        taus = [2 ** x for x in [3, 4, 5, 6]]
        print(taus)
        for tau in taus:
            print("Run SeqCoreset", tau)
            run(
                {
                    "shuffle_seed": shuffle_seed,
                    "outliers": {"Fixed": frac_out},
                    "algorithm": {"SeqCoreset": {"tau": tau}},
                    "dataset": DATASETS[dataset].get_path(),
                    "constraint": {"transversal": {"topics": constr}},
                }
            )
            print("Run StreamCoreset", tau)
            run(
                {
                    "shuffle_seed": shuffle_seed,
                    "outliers": {"Fixed": frac_out},
                    "algorithm": {"StreamingCoreset": {"tau": tau}},
                    "dataset": DATASETS[dataset].get_path(),
                    "constraint": {"transversal": {"topics": constr}},
                }
            )
        for tau in [1, 2, 4, 8]:
            for hosts in [workers[:i] for i in [2, 4, 8]]:
                print("Run MRCoreset", tau, hosts)
                # Keep the size of the final coreset constant across thread counts
                run(
                    {
                        "parallel": {"threads": 1, "hosts": hosts},
                        "shuffle_seed": shuffle_seed,
                        "outliers": {"Fixed": frac_out},
                        "algorithm": {"MapReduceCoreset": {"tau": tau}},
                        "dataset": os.path.abspath(DATASETS[dataset].get_path()),
                        "constraint": {"transversal": {"topics": constr}},
                    }
                )


def run_musixmatch():
    genres = {
        "Unknown": 6,
        "Rock": 113150,
        "Rap": 13272,
        "Latin": 6444,
        "Jazz": 17541,
        "Electronic": 30922,
        "Punk": 5652,
        "Pop": 27756,
        "New Age": 2787,
        "Metal": 11222,
        "RnB": 13208,
        "Country": 10695,
        "Reggae": 8910,
        "Folk": 6377,
        "Blues": 8125,
        "World": 4770,
    }
    # matroid of rank 273, obtained by as the quotient of the genres count by 100
    highrank_matroid = {
        "Rock": 113,
        "Rap": 13,
        "Latin": 6,
        "Jazz": 17,
        "Electronic": 30,
        "Punk": 5,
        "Pop": 27,
        "New Age": 2,
        "Metal": 11,
        "RnB": 13,
        "Country": 10,
        "Reggae": 8,
        "Folk": 6,
        "Blues": 8,
        "World": 4,
    }
    # Half of the highrank matroid for each category
    midrank_matroid = {
        "Rock": 56,
        "Rap": 6,
        "Latin": 3,
        "Jazz": 8,
        "Electronic": 15,
        "Punk": 2,
        "Pop": 13,
        "New Age": 1,
        "Metal": 5,
        "RnB": 6,
        "Country": 5,
        "Reggae": 4,
        "Folk": 3,
        "Blues": 4,
        "World": 2,
    }
    lowrank_matroid = {
        "Rock": 10,
        "Pop": 10,
    }

    datasets = ["MusixMatch", "MusixMatch-s10000"]
    for dataset in datasets:
        DATASETS[dataset].try_download_preprocessed()
        DATASETS[dataset].preprocess()
    constraints = [midrank_matroid, lowrank_matroid]
    # Fraction of allowed outliers
    frac_outliers = [10]
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
                    "outliers": {"Fixed": frac_out},
                    "algorithm": {"Random": {"seed": seed}},
                    "dataset": DATASETS[dataset].get_path(),
                    "constraint": {"partition": {"categories": constr}},
                }
            )

        # # Run the baseline algorithm
        if dataset == "MusixMatch-s10000":
            run(
                {
                    "shuffle_seed": shuffle_seed,
                    "outliers": {"Fixed": frac_out},
                    "algorithm": "ChenEtAl",
                    "dataset": DATASETS[dataset].get_path(),
                    "constraint": {"partition": {"categories": constr}},
                }
            )

        # # Run coreset algorithms
        taus = [2 ** x for x in [3, 4, 5, 6]]
        print(taus)
        for tau in taus:
            print("Run SeqCoreset", tau)
            run(
                {
                    "shuffle_seed": shuffle_seed,
                    "outliers": {"Fixed": frac_out},
                    "algorithm": {"SeqCoreset": {"tau": tau}},
                    "dataset": DATASETS[dataset].get_path(),
                    "constraint": {"partition": {"categories": constr}},
                }
            )
            print("Run StreamCoreset", tau)
            run(
                {
                    "shuffle_seed": shuffle_seed,
                    "outliers": {"Fixed": frac_out},
                    "algorithm": {"StreamingCoreset": {"tau": tau}},
                    "dataset": DATASETS[dataset].get_path(),
                    "constraint": {"partition": {"categories": constr}},
                }
            )
        for tau in [1, 2, 4, 8]:
            for hosts in [workers[:i] for i in [2, 4, 8]]:
                print("Run MRCoreset", tau, hosts)
                # Keep the size of the final coreset constant across thread counts
                run(
                    {
                        "parallel": {"threads": 1, "hosts": hosts},
                        "shuffle_seed": shuffle_seed,
                        "outliers": {"Fixed": frac_out},
                        "algorithm": {"MapReduceCoreset": {"tau": tau}},
                        "dataset": os.path.abspath(DATASETS[dataset].get_path()),
                        "constraint": {"partition": {"categories": constr}},
                    }
                )


def run_random():
    highrank_matroid = {
        "0": 1,
        "1": 1,
        "2": 1,
        "3": 1,
        "4": 1,
        "5": 1,
        "6": 1,
        "7": 1,
        "8": 1,
        "9": 1,
    }

    datasets = ["random-10000"]
    for dataset in datasets:
        DATASETS[dataset].try_download_preprocessed()
        DATASETS[dataset].preprocess()
    constraints = [highrank_matroid]
    # number of allowed outliers
    num_outliers = [10]
    # These seeds also define the number of repetitions
    shuffle_seeds = [43234, 23562, 12451, 445234, 234524]

    for shuffle_seed, dataset, constr, num_outliers in itertools.product(
        shuffle_seeds, datasets, constraints, num_outliers
    ):
        # Run the naive baseline
        print("Run random")
        for seed in [1458, 345, 65623]:
            run(
                {
                    "shuffle_seed": shuffle_seed,
                    "outliers": {"Fixed": num_outliers},
                    "algorithm": {"Random": {"seed": seed}},
                    "dataset": DATASETS[dataset].get_path(),
                    "constraint": {"partition": {"categories": constr}},
                }
            )

        # # Run the baseline algorithm
        # if dataset == "random-s10000":
        run(
            {
                "shuffle_seed": shuffle_seed,
                "outliers": {"Fixed": num_outliers},
                "algorithm": "ChenEtAl",
                "dataset": DATASETS[dataset].get_path(),
                "constraint": {"partition": {"categories": constr}},
            }
        )

        # # Run coreset algorithms
        taus = [2 ** x for x in [2, 3, 4, 5, 6]]
        print(taus)
        for tau in taus:
            print("Run SeqCoreset", tau)
            run(
                {
                    "shuffle_seed": shuffle_seed,
                    "outliers": {"Fixed": num_outliers},
                    "algorithm": {"SeqCoreset": {"tau": tau}},
                    "dataset": DATASETS[dataset].get_path(),
                    "constraint": {"partition": {"categories": constr}},
                }
            )
            print("Run StreamCoreset", tau)
            run(
                {
                    "shuffle_seed": shuffle_seed,
                    "outliers": {"Fixed": num_outliers},
                    "algorithm": {"StreamingCoreset": {"tau": tau}},
                    "dataset": DATASETS[dataset].get_path(),
                    "constraint": {"partition": {"categories": constr}},
                }
            )
        for tau in [1, 2, 4, 8]:
            for hosts in [workers[:i] for i in [2, 4, 8]]:
                print("Run MRCoreset", tau, hosts)
                # Keep the size of the final coreset constant across thread counts
                run(
                    {
                        "parallel": {"threads": 1, "hosts": hosts},
                        "shuffle_seed": shuffle_seed,
                        "outliers": {"Fixed": num_outliers},
                        "algorithm": {"MapReduceCoreset": {"tau": tau}},
                        "dataset": os.path.abspath(DATASETS[dataset].get_path()),
                        "constraint": {"partition": {"categories": constr}},
                    }
                )


def check():
    datasets = ["wiki-d10-c20"]
    # datasets = ["MusixMatch"]
    # datasets = ["random-100000"]
    for dataset in datasets:
        DATASETS[dataset].try_download_preprocessed()
        DATASETS[dataset].preprocess()
    # These seeds also define the number of repetitions
    shuffle_seeds = [
        43234,
        23562,
        12451,
        445234,
        234524,
        2346,
        3256209,
        23462,
        24572,
        476467,
        34673,
        346987,
        235,
        467858,
        246734,
        467,
    ]
    # shuffle_seeds.extend(range(1, 100))

    constraint = {"transversal": {"topics": list(range(0, 20))}}
    # constraint = {"partition": {"categories": {"signal": 10, "background": 10}}}
    outliers = {"Percentage": 0.001}

    for shuffle_seed, dataset in itertools.product(shuffle_seeds, datasets):
        for seed in [1458, 345, 65623]:
            run(
                {
                    "shuffle_seed": shuffle_seed,
                    "outliers": outliers,
                    "algorithm": {"Random": {"seed": seed}},
                    "dataset": DATASETS[dataset].get_path(),
                    "constraint": constraint,
                }
            )

        # Run coreset algorithms
        taus = [8, 32]
        for tau in taus:
            run(
                {
                    "shuffle_seed": shuffle_seed,
                    "outliers": outliers,
                    "algorithm": {"SeqCoreset": {"tau": tau}},
                    "dataset": DATASETS[dataset].get_path(),
                    "constraint": constraint,
                }
            )


if __name__ == "__main__":
    subprocess.run(["cargo", "build", "--release"])
    # check()
    # run_wiki()
    # run_musixmatch()
    run_higgs()
    # run_random()
