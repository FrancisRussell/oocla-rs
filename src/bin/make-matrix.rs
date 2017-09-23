extern crate ooc;

use std::env;
use std::path::Path;
use ooc::dense_matrix::Dense;

fn main() {
    let args : Vec<String> = env::args().collect();
    //let output = &args[1];
    let mut matrix: Dense<f32> = Dense::create(Path::new("/tmp/test.mat"), 1000, 1000).unwrap();
    matrix.randomise();
}
