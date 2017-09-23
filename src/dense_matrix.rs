use std::path::Path;
use std::io;
use std::fs::{File, OpenOptions};
use nix::sys::mman::{MapFlags, ProtFlags, MAP_SHARED, PROT_READ, PROT_WRITE, mmap, munmap};
use nix::libc::{c_void, size_t};
use std::os::unix::io::AsRawFd;
use std::error::Error;
use std::{mem, ptr, slice};
use std::marker::PhantomData;
use rand::{self, Rand, Rng};

const HEADER_SIZE: usize = 64;

#[derive(Clone, Copy)]
#[repr(C)]
pub enum FloatType {
    Single,
    Double,
}

pub trait SupportedType: Copy {
    fn get_float_type() -> FloatType;
}

impl SupportedType for f32 {
    fn get_float_type() -> FloatType {
        FloatType::Single
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

impl MatrixHeader {
    fn get_data_length_elements(&self) -> u64 {
        self.lda * if self.transposed {
            self.num_rows
        } else {
            self.num_cols
        }
    }
}

pub struct Dense<T> {
    file: File,
    start: *mut c_void,
    header: *mut MatrixHeader,
    data: *mut T,
}

impl<T> Dense<T> {
    pub fn create(path: &Path, rows: u64, cols: u64) -> Result<Dense<T>, Box<Error>> where T: SupportedType {
        let file = OpenOptions::new().read(true).write(true).create(true).open(path)?;
        let len = Self::compute_length(rows, cols);
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
        let result = Dense {
            file: file,
            start: start,
            header: header,
            data: data as *mut T,
        };
        {
            let header = result.get_header_mut();
            header.num_rows = rows;
            header.num_cols = cols;
            header.representation = T::get_float_type();
            header.transposed = false;
            header.lda = cols;
        }
        Ok(result)
    }

    pub fn num_rows(&self) -> u64 {
        self.get_header().num_rows
    }

    pub fn num_cols(&self) -> u64 {
        self.get_header().num_cols
    }

    pub fn transpose(&mut self) {
        let header = self.get_header_mut();
        header.transposed ^= true;
        mem::swap(&mut header.num_rows, &mut header.num_cols);
    }

    fn compute_length(rows: u64, cols: u64) -> u64 {
        HEADER_SIZE as u64 + rows * cols * mem::size_of::<T>() as u64
    }

    fn get_header(&self) -> &MatrixHeader {
        unsafe {
            self.header.as_ref().unwrap()
        }
    }

    fn get_data(&self) -> *const T {
        self.data
    }

    fn get_data_mut(& mut self) -> *mut T {
        self.data
    }


    fn get_header_mut(&self) -> &mut MatrixHeader {
        unsafe {
            self.header.as_mut().unwrap()
        }
    }

    fn create_index_generator(&self) -> ElementIterCommon {
        let header = self.get_header();
        let (mut major_size, mut minor_size) = (header.num_rows as usize, header.num_cols as usize);
        if header.transposed {
            mem::swap(&mut major_size, &mut minor_size);
        }
        ElementIterCommon {
            major_size: major_size as usize,
            minor_size: minor_size as usize,
            major_index: 0,
            major_offset: 0,
            minor_offset: 0,
            lda: header.lda as usize,
            transposed: header.transposed,
        }
    }

    pub fn element_iter<'a>(&'a self) -> ElementIter<'a, T> {
        let generator = self.create_index_generator();
        ElementIter {
            lifetime: PhantomData,
            generator: generator,
            data: self.get_data(),
        }
    }

    pub fn element_iter_mut<'a>(&'a mut self) -> ElementIterMut<'a, T> {
        let generator = self.create_index_generator();
        ElementIterMut {
            lifetime: PhantomData,
            generator: generator,
            data: self.get_data_mut(),
        }
    }

    pub fn randomise(&mut self) where T: Rand {
        let mut rng = rand::thread_rng();
        for value in self.element_iter_mut() {
            *value = rng.gen()
        }
    }
}

pub struct ElementIterCommon {
    major_size: usize,
    minor_size: usize,
    major_index: usize,
    major_offset: usize,
    minor_offset: usize,
    lda: usize,
    transposed: bool,
}

impl ElementIterCommon {
    fn next_index(&mut self) -> Option<usize> {
        if self.minor_offset + 1 < self.minor_size {
            self.minor_offset += 1
        } else {
            self.minor_offset = 0;
            self.major_index += 1;
            self.major_offset += self.lda;
        }
        if self.major_index < self.major_size {
            Some(self.major_offset + self.minor_offset)
        } else {
            None
        }
    }

    pub fn get_row(&self) -> usize {
        if self.transposed {
            self.minor_offset
        } else {
            self.major_index
        }
    }

    pub fn get_col(&self) -> usize {
        if self.transposed {
            self.major_index
        } else {
            self.minor_offset
        }
    }
}

pub struct ElementIter<'a, T> where T: 'a {
    lifetime: PhantomData<&'a T>,
    generator: ElementIterCommon,
    data: *const T,
}

impl <'a, T> Iterator for ElementIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        match self.generator.next_index() {
            None => None,
            Some(idx) => unsafe { self.data.offset(idx as isize).as_ref() },
        }
    }
}

impl <'a, T> ElementIter<'a, T> {
    pub fn get_row(&self) -> usize {
        self.generator.get_row()
    }

    pub fn get_col(&self) -> usize {
        self.generator.get_col()
    }
}

pub struct ElementIterMut<'a, T> where T: 'a {
    lifetime: PhantomData<&'a T>,
    data: *mut T,
    generator: ElementIterCommon,
}

impl <'a, T> Iterator for ElementIterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<&'a mut T> {
        let result:Option<&'a mut T> = match self.generator.next_index() {
            None => None,
            Some(idx) => unsafe { self.data.offset(idx as isize).as_mut() },
        };
        result
    }
}

impl <'a, T> ElementIterMut<'a, T> {
    pub fn get_row(&self) -> usize {
        self.generator.get_row()
    }

    pub fn get_col(&self) -> usize {
        self.generator.get_col()
    }
}

impl<T> Drop for Dense<T> {
    fn drop(&mut self) {
        let header = self.get_header();
        let length = Self::compute_length(header.num_rows, header.num_cols);
        unsafe {
            munmap(self.start, length as usize)
        }.unwrap();
    }
}
