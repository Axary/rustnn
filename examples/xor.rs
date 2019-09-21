extern crate rnn;

use rnn::Network;

const TRUTH_TABLE: [([f32; 2], f32); 4] = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
];

fn xor_check(network: &Network) -> f32 {
    TRUTH_TABLE.iter().fold(0.0, |total, &(input, ideal)| {
        total + rnn::func::squared_error(&[ideal], &network.run(&input))
    })
}

fn showcase(network: &Network) {
    println!("fitness of the best survivor: {}", xor_check(network));
    println!("individual results:");

    TRUTH_TABLE.iter().for_each(|&(input, ideal)| {
        println!(
            "{:?} => ideal: {:?}, actual: {:?}",
            input,
            ideal,
            network.run(&input)[0]
        );
    })
}

fn main() {
    let generations = 1000;

    let mut environment = rnn::env::Genetic::new(&[2, 2, 1], 1000);

    let now = std::time::Instant::now();

    for _ in 0..generations {
        environment.simple_gen(xor_check);
    }

    println!(
        "time spend: {:?}",
        std::time::Instant::now().duration_since(now)
    );
    showcase(environment.get_best());
}
