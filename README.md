# MACACO: MAtroid ConstrAined Clustering with Outliers

## code organization

The rust code is split into several small crates to speed up the compilation (crates get compiled in parallel).

- `experiments` contains python code that preprocesses datasets and runs experiments
- `macaco` is the Rust crate defining the binary to run the experiments
- `macaco-base` is a Rust crate containing the basic data definitions
- `macaco-sequential` is a Rust crate containing the implementation of sequential algorithms

