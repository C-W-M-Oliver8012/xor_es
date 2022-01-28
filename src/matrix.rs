use rand::prelude::*;
use rand_distr::StandardNormal;

#[derive(Clone)]
pub struct Matrix {
    pub value: Vec<Vec<f64>>,
    pub rows: usize,
    pub columns: usize,
}

pub fn new(rows: usize, columns: usize) -> Matrix {
    let mut m: Matrix = Matrix {
        value: Vec::new(),
        rows: rows,
        columns: columns,
    };

    // init matrix
    for i in 0..rows {
        m.value.push(Vec::new());
        for _j in 0..columns {
            m.value[i].push(0.0);
        }
    }

    // returns m
    m
}

pub fn new_gaussian_noise(rows: usize, columns: usize) -> Matrix {
    let mut m: Matrix = Matrix {
        value: Vec::new(),
        rows: rows,
        columns: columns,
    };

    // init matrix
    for i in 0..rows {
        m.value.push(Vec::new());
        for _j in 0..columns {
            m.value[i].push(thread_rng().sample(StandardNormal));
        }
    }

    // returns m
    m
}

pub fn print(m: &Matrix) {
    println!("[");
    for i in 0..m.rows {
        print!("[");
        for j in 0..m.columns {
            print!("{}, ", m.value[i][j]);
        }
        println!("]");
    }
    println!("]");
}

pub fn set_value(m: &mut Matrix, value: Vec<Vec<f64>>) -> Result<String, String> {
    // checks value parameters to make sure it matches matrix
    if value.len() == m.rows {
        for i in 0..m.rows {
            if value[i].len() != m.columns {
                return Err(String::from("Matrix sizes do not match."));
            }
        }
    } else {
        return Err(String::from("Matrix sizes do not match."));
    }

    for i in 0..m.rows {
        for j in 0..m.columns {
            m.value[i][j] = value[i][j];
        }
    }

    Ok(String::from("Success"))
}

pub fn multiply(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    if a.columns != b.rows {
        return Err(String::from("Matrix sizes do not match."));
    }
    let mut c: Matrix = new(a.rows, b.columns);
    for i in 0..a.rows {
        for j in 0..b.columns {
            let mut sum: f64 = 0.0;
            for k in 0..a.columns {
                sum += a.value[i][k] * b.value[k][j];
            }
            c.value[i][j] = sum;
        }
    }
    Ok(c)
}

pub fn scalar(a: &Matrix, s: f64) -> Matrix {
    let mut b = a.clone();
    for i in 0..b.rows {
        for j in 0..b.columns {
            b.value[i][j] *= s;
        }
    }
    b
}

pub fn add(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    if a.rows != b.rows || a.columns != b.columns {
        return Err(String::from("Matrix sizes do not match."));
    }
    let mut c: Matrix = new(a.rows, a.columns);
    for i in 0..a.rows {
        for j in 0..a.columns {
            c.value[i][j] = a.value[i][j] + b.value[i][j];
        }
    }
    Ok(c)
}

pub fn subtract(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    if a.rows != b.rows || a.columns != b.columns {
        return Err(String::from("Matrix sizes do not match."));
    }
    let mut c: Matrix = new(a.rows, a.columns);
    for i in 0..a.rows {
        for j in 0..a.columns {
            c.value[i][j] = a.value[i][j] - b.value[i][j];
        }
    }
    Ok(c)
}

pub fn mean(m: &Matrix) -> f64 {
    let mut sum: f64 = 0.0;
    for i in 0..m.rows {
        for j in 0..m.columns {
            sum += m.value[i][j];
        }
    }
    sum / (m.rows * m.columns) as f64
}

pub fn variance(m: &Matrix, mean: f64) -> f64 {
    let mut v: f64 = 0.0;
    for i in 0..m.rows {
        for j in 0..m.columns {
            v += (m.value[i][j] - mean).powi(2);
        }
    }
    v / (m.rows * m.columns) as f64
}

pub fn normalize(a: &Matrix) -> Matrix {
    let mean = mean(a);
    let variance = variance(a, mean);
    let standard_deviation = variance.sqrt();
    let mean_diff = 0.0 - mean;

    let mut b = a.clone();
    for i in 0..b.rows {
        for j in 0..b.columns {
            b.value[i][j] += mean_diff;
            if standard_deviation != 0.0 {
                b.value[i][j] /= standard_deviation;
            }
        }
    }
    b
}

pub fn activate(a: &Matrix) -> Matrix {
    let mut b = a.clone();
    for i in 0..b.rows {
        for j in 0..b.columns {
            b.value[i][j] = leaky_relu(b.value[i][j]);
        }
    }
    b
}

pub fn leaky_relu(v: f64) -> f64 {
    if v >= 0.0 {
        return v;
    } else {
        return 0.25 * v;
    }
}
