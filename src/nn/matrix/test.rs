#[cfg(test)]
use crate::matrix;

#[test]
fn new_test() {
    let a = matrix::new(2, 3);

    assert_eq!(a.rows, 2);
    assert_eq!(a.columns, 3);
    assert_eq!(a.value, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn new_gaussian_noise_test() {
    let a = matrix::new_gaussian_noise(2, 3);

    assert_eq!(a.rows, 2);
    assert_eq!(a.columns, 3);
    assert_ne!(a.value, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn print_test() {
    let a = matrix::new(2, 2);
    // pass = does not panic
    matrix::print(&a);
}

#[test]
fn multiply_test() {
    let mut a = matrix::new(2, 3);
    let mut b = matrix::new(3, 3);
    a.value = vec![4.0, 6.0, 1.0, 9.0, 7.0, 3.0];
    b.value = vec![2.0, 8.0, 3.0, 3.0, 2.0, 8.0, 7.0, 2.0, 9.0];

    let c = matrix::multiply(&a, &b).unwrap();
    assert_eq!(c.rows, 2);
    assert_eq!(c.columns, 3);
    assert_eq!(c.value, [37.0, 93.0, 70.0, 60.0, 93.0, 87.0]);
}

#[test]
#[should_panic]
fn multiply_panic_test() {
    let mut a = matrix::new(2, 3);
    let mut b = matrix::new(3, 3);
    a.value = vec![4.0, 6.0, 1.0, 9.0, 7.0, 3.0];
    b.value = vec![2.0, 8.0, 3.0, 3.0, 2.0, 8.0, 7.0, 2.0, 9.0];
    let _ = matrix::multiply(&b, &a).unwrap();
}

#[test]
fn add_test() {
    let mut a = matrix::new(2, 3);
    let mut b = matrix::new(2, 3);
    a.value = vec![4.0, 6.0, 1.0, 9.0, 7.0, 3.0];
    b.value = vec![2.0, 8.0, 3.0, 3.0, 2.0, 8.0];

    let c = matrix::add(&a, &b).unwrap();
    assert_eq!(c.rows, 2);
    assert_eq!(c.columns, 3);
    assert_eq!(c.value, [6.0, 14.0, 4.0, 12.0, 9.0, 11.0]);
}

#[test]
#[should_panic]
fn add_panic_test() {
    let mut a = matrix::new(2, 3);
    let mut b = matrix::new(2, 2);
    a.value = vec![4.0, 6.0, 1.0, 9.0, 7.0, 3.0];
    b.value = vec![2.0, 8.0, 3.0, 3.0];

    let _ = matrix::add(&a, &b).unwrap();
}

#[test]
fn scalar_test() {
    let mut a = matrix::new(2, 3);
    a.value = vec![4.0, 6.0, 1.0, 9.0, 7.0, 3.0];

    let b = matrix::scalar(&a, 1.3);
    assert_eq!(b.rows, 2);
    assert_eq!(b.columns, 3);
    assert_eq!(b.value, [4.0 * 1.3, 6.0 * 1.3, 1.0 * 1.3, 9.0 * 1.3, 7.0 * 1.3, 3.0 * 1.3]);
}

#[test]
fn mean_test() {
    let mut a = matrix::new(2, 2);
    a.value = vec![2.0, 4.0, 1.0, 7.0];

    assert_eq!(matrix::mean(&a), 3.5);
}

#[test]
fn variance_test() {
   let mut a = matrix::new(2, 2);
   a.value = vec![2.0, 4.0, 1.0, 7.0];

   assert_eq!(matrix::variance(&a, matrix::mean(&a)), 5.25);
}

#[test]
fn normalize_test() {
    let mut a = matrix::new(2, 2);
    a.value = vec![2.0, 4.0, 1.0, 7.0];

    let mean = matrix::mean(&a);
    let standard_deviation = matrix::variance(&a, mean).sqrt();
    let expected_b_value = [
        (a.value[0] - mean) / standard_deviation,
        (a.value[1] - mean) / standard_deviation,
        (a.value[2] - mean) / standard_deviation,
        (a.value[3] - mean) / standard_deviation,
    ];

    let b = matrix::normalize(&a);
    assert_eq!(b.rows, 2);
    assert_eq!(b.columns, 2);
    assert_eq!(b.value, expected_b_value);
    assert_eq!(matrix::mean(&b), 0.0);
    assert_eq!(matrix::variance(&b, matrix::mean(&b)).sqrt(), 1.0);
}

#[test]
fn leaky_relu_test() {
    let mut a = matrix::new(2, 2);
    a.value = vec![2.0, -4.0, -1.0, 7.0];

    let b = matrix::leaky_relu(&a);
    assert_eq!(b.rows, 2);
    assert_eq!(b.columns, 2);
    assert_eq!(b.value, [2.0, -0.04, -0.01, 7.0]);
}
