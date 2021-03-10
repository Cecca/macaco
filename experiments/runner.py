import json
import base64
import subprocess


EXECUTABLE = "target/release/kcmkc"


def run(configuration):
    conf_str = base64.b64encode(json.dumps(configuration).encode("utf-8"))
    subprocess.run([EXECUTABLE, conf_str])


run(
    {
        "outliers": {"Fixed": 10},
        "algorithm": {"Random": {"seed": 2145}},
        "dataset": ".datasets/sampled/Wikipedia-date-20210120-dimensions-50-topics-100-sample10000-v1.msgpack.gz",
        "constraint": {"transversal": {"topics": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}},
    }
)
