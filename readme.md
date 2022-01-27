# What is this?
This is a proof of concept implementation of the algorithm described in this
[OpenAI Blogpost](https://openai.com/blog/evolution-strategies/) and also in this [study](https://arxiv.org/pdf/1703.03864.pdf).

# How does the program work?
The program is a rust project made up of three main components: matrix.rs, nn.rs, and main.rs. matrix.rs contains a small and simple matrix library built simply for use in nn.rs which is a small feedforward neural network library. These libraries are far from production quality but they do work for a simply project like this.

# How do I run the program?
1. [Install Rust](https://www.rust-lang.org/)
2. The program is multithreaded so it is wise to adust the constant named NT in the main.rs file to match the number of threads that your computer can run (4 is probably a safe bet on most newer computers).
3. Go to the project directory in a terminal and run "cargo run --release"

# What does the output mean?
The program usually runs and finishes quite quickly (less than 10 seconds on my computer), however, this is dependent on the computer and how "far" from the solution the neural network randomly initializes to. In the output, you may see something like "Generation 500: -0.541232132123." The program works by generating a population of mutated neural networks, and then approximates the gradient and updated the neural network between every generation. The decimal number is the result of the reward function, which we are trying to optimize. Zero is the highest result possible. The program stops running when the reward program returns a reward greater than -0.1.

# Is there any significance to this project?
No. This project was meant for learning and proof of concept. There really is not anything special about solving xor with any method. In the future, I attempt to use this project as the steeping stone to more complicated projects.
