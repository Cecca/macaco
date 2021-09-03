# k-center clustering under matroid constraints

## code organization

The rust code is split into several small crates to speed up the compilation (crates get compiled in parallel).

- `experiments` contains python code that preprocesses datasets and runs experiments
- `kcmkc` is the Rust crate defining the binary to run the experiments
- `kcmkc-base` is a Rust crate containing the basic data definitions
- `kcmkc-sequential` is a Rust crate containing the implementation of sequential algorithms

## TODO

- [ ] run experiments with fewer outliers (like 50/100 for each dataset)
- [ ] analyze the ratio between the time to compute the coreset and the time to compute the solution
- [ ] analyze the memory usage of the streaming algorithm, compared with the sequential algorithm
