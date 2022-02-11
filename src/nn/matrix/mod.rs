pub mod test;

extern crate blas_src;

use rand::prelude::*;
use rand_distr::StandardNormal;
use blas;

#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub columns: usize,
    pub value: Vec<f64>,
}

pub fn new(rows: usize, columns: usize) -> Matrix {
    let mut a = Matrix {
        rows: rows,
        columns: columns,
        value: Vec::with_capacity(rows * columns),
    };

    for _ in 0..a.rows * a.columns {
        a.value.push(0.0);
    }
    a
}

pub fn new_gaussian_noise(rows: usize, columns: usize) -> Matrix {
    let mut a = Matrix {
        rows: rows,
        columns: columns,
        value: Vec::with_capacity(rows * columns),
    };

    for _ in 0..a.rows * a.columns {
        a.value.push(thread_rng().sample(StandardNormal));
    }
    a
}

pub fn print(a: &Matrix) {
    print!("[");
    for i in 0..a.rows {
        print!("[");
        for j in 0..a.columns {
            let index = j * a.rows + i;
            print!("{}, ", a.value[index]);
        }
        if i == a.rows - 1 {
            print!("]");
        } else {
            println!("],");
        }
    }
    println!("]");
    println!();
}

pub fn multiply(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    if a.columns != b.rows {
        return Err(String::from("Matrix sizes are incorrect."));
    }
    let mut c = new(a.rows, b.columns);
    let (m, n, k) = (a.rows, b.columns, a.columns);

    unsafe {
        blas::dgemm(
            'N' as u8,
            'N' as u8,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            &a.value,
            m as i32,
            &b.value,
            k as i32,
            0.0,
            &mut c.value,
            m as i32
        );
    }
    Ok(c)
}

pub fn add(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    if a.rows != b.rows || a.columns != b.columns {
        return Err(String::from("Matrix sizes are incorrect."));
    }

    let mut c = new(a.rows, a.columns);
    for i in 0..c.rows * c.columns {
        c.value[i] = a.value[i] + b.value[i];
    }
    Ok(c)
}

pub fn scalar(a: &Matrix, s: f64) -> Matrix {
    let mut b = a.clone();
    unsafe {
        blas::dscal((b.rows * b.columns) as i32, s, &mut b.value, 1);
    }
    b
}

pub fn mean(a: &Matrix) -> f64 {
    let mut sum: f64 = 0.0;
    for i in 0..a.rows * a.columns {
        sum += a.value[i];
    }
    sum / (a.rows * a.columns) as f64
}

pub fn variance(a: &Matrix, mean: f64) -> f64 {
    let mut variance: f64 = 0.0;
    for i in 0..a.rows * a.columns {
        variance += (a.value[i] - mean).powi(2);
    }
    variance / (a.rows * a.columns) as f64
}

pub fn normalize(a: &Matrix) -> Matrix {
    let mean = mean(a);
    let variance = variance(a, mean);
    let standard_deviation = variance.sqrt();
    let mean_diff = 0.0 - mean;

    let mut b = a.clone();
    for i in 0..b.rows * b.columns {
        b.value[i] += mean_diff;
        if standard_deviation != 0.0 {
            b.value[i] /= standard_deviation;
        }
    }
    b
}

pub fn leaky_relu(a: &Matrix) -> Matrix {
    let mut b = a.clone();
    for i in 0..b.rows * b.columns {
        if b.value[i] < 0.0 {
            b.value[i] *= 0.01;
        }
    }
    b
}
