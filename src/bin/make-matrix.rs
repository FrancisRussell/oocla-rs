extern crate ooc;

use std::env;
use std::path::Path;
use ooc::disk_matrix::{DiskMatrix, FloatType};

fn main() {
    let args : Vec<String> = env::args().collect();
    //let output = &args[1];
    let matrix = DiskMatrix::create(Path::new("/tmp/test.mat"), 1000, 1000, FloatType::Double);
}
