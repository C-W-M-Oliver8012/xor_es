use crate::matrix;

#[derive(Clone)]
pub struct NN {
    pub topology: Vec<usize>,
    pub connections: Vec<matrix::Matrix>,
    pub biases: Vec<matrix::Matrix>
}

pub fn new(topology: Vec<usize>) -> NN {
    let mut nn = NN {
        topology: topology.clone(),
        connections: Vec::new(),
        biases: Vec::new(),
    };

    // create connections
    for i in 1..nn.topology.len() {
        nn.connections.push(matrix::new(nn.topology[i - 1], nn.topology[i]));
    }

    // create biases
    for i in 1..nn.topology.len() {
        nn.biases.push(matrix::new(1, nn.topology[i]));
    }

    nn
}

pub fn new_gaussian_noise(topology: Vec<usize>) -> NN {
    let mut nn = NN {
        topology: topology.clone(),
        connections: Vec::new(),
        biases: Vec::new(),
    };

    // create connections
    for i in 1..nn.topology.len() {
        nn.connections.push(matrix::new_gaussian_noise(nn.topology[i - 1], nn.topology[i]));
    }

    // create biases
    for i in 1..nn.topology.len() {
        nn.biases.push(matrix::new_gaussian_noise(1, nn.topology[i]));
    }

    nn
}

pub fn print(nn: &NN) {
    println!("Topology:");
    print!("[");
    for i in 0..nn.topology.len() {
        print!("{}, ", nn.topology[i]);
    }
    print!("]");
    println!();
    println!();

    println!("Connections:");
    for i in 0..nn.connections.len() {
        println!("{}x{}", nn.connections[i].rows, nn.connections[i].columns);
        matrix::print(&nn.connections[i]);
        println!();
    }
    println!();

    println!("Biases:");
    for i in 0..nn.biases.len() {
        println!("{}x{}", nn.biases[i].rows, nn.biases[i].columns);
        matrix::print(&nn.biases[i]);
        println!();
    }
    println!();
}

pub fn feedforward(nn: &NN, input: &matrix::Matrix) -> Result<matrix::Matrix, String> {
    if input.columns != nn.topology[0] || input.rows != 1 {
        return Err(String::from("Input of incorrect size."));
    }

    let mut current_output = matrix::normalize(&input);

    for i in 0..nn.connections.len() {
        current_output = matrix::multiply(&current_output, &nn.connections[i]).unwrap();
        current_output = matrix::add(&current_output, &nn.biases[i]).unwrap();
        if i != nn.connections.len() - 1 {
            current_output = matrix::normalize(&current_output);
        }
        current_output = matrix::activate(&current_output);
    }
    //current_output = matrix::softmax(&current_output);

    Ok(current_output)
}

pub fn add(nn1: &NN, nn2: &NN) -> Result<NN, String> {
    if nn1.topology.len() != nn2.topology.len() {
        return Err(String::from("Topologies do not match."));
    }

    for i in 0..nn1.topology.len() {
        if nn1.topology[i] != nn2.topology[i] {
            return Err(String::from("Topologies do not match."));
        }
    }

    let mut nn3 = new(nn1.topology.clone());

    for i in 0..nn1.connections.len() {
        nn3.connections[i] = matrix::add(&nn1.connections[i], &nn2.connections[i]).unwrap();
        nn3.biases[i] = matrix::add(&nn1.biases[i], &nn2.biases[i]).unwrap();
    }

    Ok(nn3)
}

pub fn scalar(nn1: &NN, s: f64) -> NN {
    let mut nn2 = new(nn1.topology.clone());

    for i in 0..nn1.connections.len() {
        nn2.connections[i] = matrix::scalar(&nn1.connections[i], s);
        nn2.biases[i] = matrix::scalar(&nn1.biases[i], s);
    }

    nn2
}
