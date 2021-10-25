use vergen::{vergen, Config, Error};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate the default 'cargo:' instruction output
    vergen(Config::default()).map_err(Error::into)
}
