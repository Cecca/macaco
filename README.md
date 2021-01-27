# k-center clustering under matroid constraints

## code organization

The rust code is split into several small crates to speed up the compilation (crates get compiled in parallel).

- `experiments` contains python code that preprocesses datasets and runs experiments
- `kcmkc-base` is a Rust crate containing the basic data definitions
- `kcmkc-sequential` is a Rust crate containing the implementation of sequential algorithms

