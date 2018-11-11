/// the complete code used in the [README](../README.md).
extern crate rnn;

const TRUTH_TABLE: [([f32; 3], f32); 8] =
    [
        ([0.0, 0.0, 0.0], 0.0),
        ([0.0, 0.0, 1.0], 1.0),
        ([0.0, 1.0, 0.0], 0.0),
        ([0.0, 1.0, 1.0], 1.0),
        ([1.0, 0.0, 0.0], 0.0),
        ([1.0, 0.0, 1.0], 1.0),
        ([1.0, 1.0, 0.0], 1.0),
        ([1.0, 1.0, 1.0], 0.0),
    ];

fn loss_function(network: &rnn::Network) -> f32 {    
    use rnn::func::squared_error;

    TRUTH_TABLE.iter().fold(0.0, |total, &(input, ideal)| {
        total + squared_error(&[ideal], &network.run(&input))
    })
}

fn showcase(network: &rnn::Network) {    
    println!("fitness of the best survivor: {}", loss_function(network));
    println!("individual results:");

    TRUTH_TABLE.iter().for_each(|&(input, ideal)| {
        println!("{:?} => ideal: {:?}, actual: {:?}", input, ideal, network.run(&input)[0]);
    })
}

fn main() {
    let network_count = 250;
    // 3 input nodes, 1 hidden layer with 3 nodes and one output node
    let layers = [3, 3, 1];

    let mut environment = rnn::env::Genetic::new(&layers, network_count);

    // run 1000 generations
    for _ in 0..1000 {
        environment.simple_gen(loss_function);
    }

    showcase(environment.get_best());
}