pub mod nn;
pub mod matrix;

use std::sync::mpsc;
use std::thread;

// learning rate
const LR: f64 = 0.01;
// standard deviation
const SD: f64 = 0.1;
// population size
const PS: usize = 100;
// num threads
const NT: usize = 8;

fn score(nn: &nn::NN) -> (f64, bool) {
    let mut pass: bool = true;
    let mut score: f64 = 0.0;
    let mut input = matrix::new(1, 2);

    // output.value[0][0] = 0.0
    // output.value[0][1] = 1.0

    // 0.0, 0.0 -> 0.0
    matrix::set_value(&mut input, vec![vec![0.0, 0.0]]).unwrap();
    let output = nn::feedforward(&nn, &input).unwrap();
    score += output.value[0][0] - output.value[0][1];
    if output.value[0][0] <= output.value[0][1] {
        pass = false;
    }

    // 1.0, 0.0 -> 1.0
    matrix::set_value(&mut input, vec![vec![1.0, 0.0]]).unwrap();
    let output = nn::feedforward(&nn, &input).unwrap();
    score += output.value[0][1] - output.value[0][0];
    if output.value[0][1] <= output.value[0][0] {
        pass = false;
    }

    // 0.0, 1.0 -> 1.0
    matrix::set_value(&mut input, vec![vec![0.0, 1.0]]).unwrap();
    let output = nn::feedforward(&nn, &input).unwrap();
    score += output.value[0][1] - output.value[0][0];
    if output.value[0][1] <= output.value[0][0] {
        pass = false;
    }

    // 1.0, 1.0 -> 0.0
    matrix::set_value(&mut input, vec![vec![1.0, 1.0]]).unwrap();
    let output = nn::feedforward(&nn, &input).unwrap();
    score += output.value[0][0] - output.value[0][1];
    if output.value[0][0] <= output.value[0][1] {
        pass = false;
    }

    (score, pass)
}

fn main() {
    // initial parameters
    let mut nn = nn::new_gaussian_noise(vec![2, 4, 4, 2]);

    // g = generation
    let mut g = 0;
    // update loop
    while score(&nn).1 == false {
        // print results every 50 generations
        if g % 1 == 0 {
            let s = score(&nn);
            println!("Generation {}: {}", g, s.0);
            println!();
        }
        // increment generation
        g += 1;

        // will be used to update the network at end of loop
        let mut update = nn::new(vec![2, 4, 4, 2]);

        // sender and receiver between threads
        let (tx, rx) = mpsc::channel();
        // stores handles of threads to join together later
        let mut handles = vec![];

        // generate num threads
        for _ in 0..NT {
            // clone sender
            let txc = tx.clone();
            // clone neural network
            let nnc = nn.clone();
            // create thread
            let handle = thread::spawn(move || {
                let mut population: Vec<nn::NN> = Vec::new();
                let mut gaussian_noise: Vec<nn::NN> = Vec::new();
                let mut pop_scores: Vec<f64> = Vec::new();
                let mut weighted_gaussian: Vec<nn::NN> = Vec::new();
                let mut thread_update = nn::new(vec![2, 4, 4, 2]);

                for i in 0..PS {
                    // init gaussian noise
                    gaussian_noise.push(nn::new_gaussian_noise(vec![2, 4, 4, 2]));
                    // multiply gaussian_noise by standard deviation
                    let temp = nn::scalar(&gaussian_noise[i], SD);
                    // add to current parameters
                    population.push(nn::add(&nnc, &temp).unwrap());
                    // score new population
                    pop_scores.push(score(&population[i]).0);
                    // calc weighted_gaussian
                    weighted_gaussian.push(nn::scalar(&gaussian_noise[i], pop_scores[i]));
                    // add weighted_gaussian to thread_update
                    thread_update = nn::add(&thread_update, &weighted_gaussian[i]).unwrap();
                }
                // send thread_update to receiver
                txc.send(thread_update).unwrap();
            });
            // add thread handle
            handles.push(handle);
        }

        // for every handle, join back to main thread
        for handle in handles {
            handle.join().unwrap();
        }

        // drop tx otherwise receiver hangs
        drop(tx);

        // calc update
        for r in rx {
            update = nn::add(&update, &r).unwrap();
        }
        update = nn::scalar(&update, LR / ((PS * NT) as f64 * SD));

        // update parameters
        nn = nn::add(&nn, &update).unwrap();
    }
    // print final generation and neural network topology
    let s = score(&nn);
    println!("Generation {}: {}", g, s.0);
    nn::print(&nn);
    println!();

    let mut input = matrix::new(1, 2);
    input.value = vec![vec![0.0, 0.0]];
    let output = nn::feedforward(&nn, &input).unwrap();
    println!("0.0, 0.0 = {}, {}", output.value[0][0], output.value[0][1]);

    input.value = vec![vec![1.0, 0.0]];
    let output = nn::feedforward(&nn, &input).unwrap();
    println!("1.0, 0.0 = {}, {}", output.value[0][0], output.value[0][1]);

    input.value = vec![vec![0.0, 1.0]];
    let output = nn::feedforward(&nn, &input).unwrap();
    println!("0.0, 1.0 = {}, {}", output.value[0][0], output.value[0][1]);

    input.value = vec![vec![1.0, 1.0]];
    let output = nn::feedforward(&nn, &input).unwrap();
    println!("1.0, 1.0 = {}, {}", output.value[0][0], output.value[0][1]);
}
