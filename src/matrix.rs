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

#[cfg(test)]
mod tests {
    use crate::matrix;

    #[test]
    fn new_test() {
        let tm = matrix::new(2, 5);
        assert_eq!(tm.rows, 2);
        assert_eq!(tm.columns, 5);
        assert_eq!(tm.value, [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]);
    }

    #[test]
    fn new_gaussian_noise_test() {
        let tm = matrix::new_gaussian_noise(2, 5);
        assert_eq!(tm.rows, 2);
        assert_eq!(tm.columns, 5);
        assert_ne!(tm.value, [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]);
    }

    #[test]
    fn print_test() {
        let tm = matrix::new(2, 5);
        matrix::print(&tm);
    }

    #[test]
    fn set_value_test() {
        let mut tm = matrix::new(2, 5);
        matrix::set_value(&mut tm, vec![
                vec![2.0, 4.0, 6.7, 1.78, 4.357],
                vec![5.7, 8.94, 2.63, 2.56, 3.94]
        ]).unwrap();
        assert_eq!(tm.value, vec![
                vec![2.0, 4.0, 6.7, 1.78, 4.357],
                vec![5.7, 8.94, 2.63, 2.56, 3.94]
        ]);
    }

    #[test]
    fn multiply_test() {
        let mut a = matrix::new(2, 5);
        let mut b = matrix::new(5, 2);

        a.value = vec![
            vec![1.0, 3.0, 6.0, 7.0, 2.0],
            vec![12.0, 4.0, 10.0, 5.0, 3.0]
        ];

        b.value = vec![
            vec![3.0, 7.0],
            vec![1.0, 6.0],
            vec![8.0, 20.0],
            vec![3.0, 8.0],
            vec![9.0, 2.0]
        ];

        let c = matrix::multiply(&a, &b).unwrap();
        assert_eq!(c.rows, 2);
        assert_eq!(c.columns, 2);
        assert_eq!(c.value, [[93.0, 205.0], [162.0, 354.0]]);
    }

    #[test]
    fn scalar_test() {
        let mut a = matrix::new(2, 5);
        a.value = vec![
            vec![1.0, 3.0, 6.0, 7.0, 2.0],
            vec![12.0, 4.0, 10.0, 5.0, 3.0]
        ];

        let b = matrix::scalar(&a, 2.0);
        assert_eq!(b.rows, 2);
        assert_eq!(b.columns, 5);
        assert_eq!(b.value, [[2.0, 6.0, 12.0, 14.0, 4.0], [24.0, 8.0, 20.0, 10.0, 6.0]]);
    }

    #[test]
    fn add_test() {
        let mut a = matrix::new(2, 2);
        let mut b = matrix::new(2, 2);

        a.value = vec![
            vec![2.0, 4.0],
            vec![1.0, 7.0]
        ];

        b.value = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0]
        ];

        let c = matrix::add(&a, &b).unwrap();
        assert_eq!(c.rows, 2);
        assert_eq!(c.columns, 2);
        assert_eq!(c.value, [[3.0, 6.0], [4.0, 11.0]]);
    }

    #[test]
    fn mean_test() {
        let mut a = matrix::new(2, 2);
        a.value = vec![
            vec![2.0, 4.0],
            vec![1.0, 7.0]
        ];

        assert_eq!(matrix::mean(&a), 3.5);
    }

    #[test]
    fn variance_test() {
        let mut a = matrix::new(2, 2);
        a.value = vec![
            vec![2.0, 4.0],
            vec![1.0, 7.0]
        ];

        assert_eq!(matrix::variance(&a, matrix::mean(&a)), 5.25);
    }

    #[test]
    fn normalize_test() {
        let mut a = matrix::new(2, 2);
        a.value = vec![
            vec![2.0, 4.0],
            vec![1.0, 7.0]
        ];

        let mean = matrix::mean(&a);
        let standard_deviation = matrix::variance(&a, mean).sqrt();
        let expected_b_value = [
            [(a.value[0][0] - mean) / standard_deviation, (a.value[0][1] - mean) / standard_deviation],
            [(a.value[1][0] - mean) / standard_deviation, (a.value[1][1] - mean) / standard_deviation],
        ];

        let b = matrix::normalize(&a);
        assert_eq!(b.rows, 2);
        assert_eq!(b.columns, 2);
        assert_eq!(b.value, expected_b_value);
        assert_eq!(matrix::mean(&b), 0.0);
        assert_eq!(matrix::variance(&b, matrix::mean(&b)).sqrt(), 1.0);
    }

    #[test]
    fn activate_test() {
        let mut a = matrix::new(2, 2);
        a.value = vec![
            vec![2.0, -4.0],
            vec![-1.0, 7.0]
        ];

        let b = matrix::activate(&a);
        assert_eq!(b.rows, 2);
        assert_eq!(b.columns, 2);
        assert_eq!(b.value, [[2.0, -1.0], [-0.25, 7.0]]);
    }

    #[test]
    fn leaky_relu_test() {
        let a: f64 = 1.45;
        let b: f64 = -8.95;

        assert_eq!(matrix::leaky_relu(a), a);
        assert_eq!(matrix::leaky_relu(b), b * 0.25);
    }
}
