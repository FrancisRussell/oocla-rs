use std::path::Path;
use std::io;
use std::fs::{File, OpenOptions};
use nix::sys::mman::{MapFlags, ProtFlags, MAP_SHARED, PROT_READ, PROT_WRITE, mmap, munmap};
use nix::libc::{c_void, size_t};
use std::os::unix::io::AsRawFd;
use std::ptr;
use std::error::Error;

const HEADER_SIZE: usize = 64;

#[derive(Clone, Copy)]
#[repr(C)]
pub enum FloatType {
    Single,
    Double,
}

impl FloatType {
    pub fn get_width(&self) -> usize {
        match *self {
            FloatType::Single => 4,
            FloatType::Double => 8,
        }
    }
}

#[repr(C)]
struct MatrixHeader {
    magic: u64,
    num_rows: u64,
    num_cols: u64,
    representation: FloatType,
    lda: u64,
    transposed: bool,
}

pub struct DiskMatrix {
    file: File,
    start: *mut c_void,
    header: *mut MatrixHeader,
    data_single: *mut f32,
    data_double: *mut f64,
}

impl DiskMatrix {
    pub fn create(path: &Path, rows: u64, cols: u64, representation: FloatType) -> Result<DiskMatrix, Box<Error>> {
        let file = OpenOptions::new().read(true).write(true).create(true).open(path)?;
        let len = Self::compute_length(rows, cols, representation);
        file.set_len(len)?;

        let mut map_flags = MapFlags::empty();
        map_flags.insert(MAP_SHARED);
        let mut prot_flags = ProtFlags::empty();
        prot_flags.insert(PROT_READ);
        prot_flags.insert(PROT_WRITE);
        let offset = 0;
        let fd = file.as_raw_fd();

        let start = unsafe {
            mmap(ptr::null_mut(), len as size_t, prot_flags, map_flags, fd, offset)
        }?;
        let header = start as *mut MatrixHeader;
        let data = unsafe {
            (start as *mut u8).offset(HEADER_SIZE as isize)
        };
        let result = DiskMatrix {
            file: file,
            start: start,
            header: header,
            data_single: data as *mut f32,
            data_double: data as *mut f64,
        };
        {
            let header = result.get_header_mut();
            header.num_rows = rows;
            header.num_cols = cols;
            header.representation = representation;
            header.transposed = false;
            header.lda = cols * representation.get_width() as u64;
        }
        Ok(result)
    }

    fn compute_length(rows: u64, cols: u64, repr: FloatType) -> u64 {
        HEADER_SIZE as u64 + rows * cols * repr.get_width() as u64
    }

    fn get_header(&self) -> &MatrixHeader {
        unsafe {
            self.header.as_ref().unwrap()
        }
    }

    fn get_header_mut(&self) -> &mut MatrixHeader {
        unsafe {
            self.header.as_mut().unwrap()
        }
    }
}

impl Drop for DiskMatrix {
    fn drop(&mut self) {
        let header = self.get_header();
        let length = Self::compute_length(header.num_rows, header.num_cols, header.representation);
        unsafe {
            munmap(self.start, length as usize)
        }.unwrap();
    }
}
