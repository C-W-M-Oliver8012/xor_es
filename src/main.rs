pub mod nn;
pub mod matrix;

use std::sync::mpsc;
use std::thread;

const LR: f64 = 0.01;
const SD: f64 = 0.1;
const PS: usize = 100;
const NT: usize = 10;

fn score(nn: &nn::NN) -> f64 {
    let mut score: f64 = 0.0;
    let mut input = matrix::new(1, 2);

    // 0.0, 0.0 -> 0.0
    matrix::set_value(&mut input, vec![vec![0.0, 0.0]]).unwrap();
    let output = nn::feedforward(&nn, &input).unwrap();
    score -= (0.0 - output.value[0][0]).abs();

    // 1.0, 0.0 -> 1.0
    matrix::set_value(&mut input, vec![vec![1.0, 0.0]]).unwrap();
    let output = nn::feedforward(&nn, &input).unwrap();
    score -= (1.0 - output.value[0][0]).abs();

    // 0.0, 1.0 -> 1.0
    matrix::set_value(&mut input, vec![vec![0.0, 1.0]]).unwrap();
    let output = nn::feedforward(&nn, &input).unwrap();
    score -= (1.0 - output.value[0][0]).abs();

    // 1.0, 1.0 -> 0.0
    matrix::set_value(&mut input, vec![vec![1.0, 1.0]]).unwrap();
    let output = nn::feedforward(&nn, &input).unwrap();
    score -= (0.0 - output.value[0][0]).abs();

    score
}

fn main() {
    // initial parameters
    let mut nn = nn::new_gaussian_noise(vec![2, 2, 1]);

    let mut g = 0;
    while score(&nn) <= -0.1 {
        if g % 50 == 0 {
            println!("Generation {}: {}", g, score(&nn));
            println!();
        }
        g += 1;

        let mut update = nn::new(vec![2, 2, 1]);

        let (tx, rx) = mpsc::channel();
        let mut handles = vec![];

        for _ in 0..NT {
            let txc = tx.clone();
            let nnc = nn.clone();
            let handle = thread::spawn(move || {
                let mut population: Vec<nn::NN> = Vec::new();
                let mut gaussian_noise: Vec<nn::NN> = Vec::new();
                let mut pop_scores: Vec<f64> = Vec::new();
                let mut weighted_gaussian: Vec<nn::NN> = Vec::new();
                let mut thread_update = nn::new(vec![2, 2, 1]);

                for i in 0..PS {
                    // init gaussian noise
                    gaussian_noise.push(nn::new_gaussian_noise(vec![2, 2, 1]));
                    // multiply gaussian_noise by standard deviation
                    let temp = nn::scalar(&gaussian_noise[i], SD);
                    // add to current parameters
                    population.push(nn::add(&nnc, &temp).unwrap());
                    // score new population
                    pop_scores.push(score(&population[i]));
                    // calc weighted_gaussian
                    weighted_gaussian.push(nn::scalar(&gaussian_noise[i], pop_scores[i]));
                    // add weighted_gaussian to thread_update
                    thread_update = nn::add(&thread_update.clone(), &weighted_gaussian[i]).unwrap();
                }
                txc.send(thread_update.clone()).unwrap();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        drop(tx);

        // calc update
        for r in rx {
            update = nn::add(&update.clone(), &r).unwrap();
        }
        update = nn::scalar(&update.clone(), LR / (PS as f64 * SD));

        // update parameters
        nn = nn::add(&nn.clone(), &update).unwrap();
    }
    println!("Generation {}: {}", g, score(&nn));
    nn::print(&nn);
    println!();
}
