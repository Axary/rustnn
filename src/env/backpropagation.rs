use crate::Network;

use rand::distributions::{Bernoulli, Distribution, Uniform};
use rand::seq::SliceRandom;
use rand::Rng;

pub struct Backpropagation {
    network: Network,
    generation: u32,
}

impl Backpropagation {
    pub fn new(network_type: &[usize]) -> Self {
        let network = Network::new(network_type);

        Self {
            network,
            generation: 0,
        }
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    pub fn train(&mut self, learning_rate: f32, input: &[f32], expected: &[f32]) {
        let output = self.network.run_remembering_state(input);

        let total_error =
            output.last().unwrap().iter().zip(expected.iter()).fold(
                0.0,
                |value, (&actual, &expected): (&f32, &f32)| {
                    value + 0.5 * (expected - actual).powi(2)
                },
            );

        // delta E_total / delta out = -(exepcted - actual)
        let output_expected = output
            .last()
            .unwrap()
            .iter()
            .zip(expected.iter())
            .map(|(actual, expected)| actual - expected)
            .collect::<Vec<_>>();
    }

    pub fn simple_gen(&mut self, learning_rate: f32, training_data: &[(Vec<f32>, Vec<f32>)]) {
        self.generation += 1;

        for (input, expected) in training_data {
            self.train(learning_rate, input, expected);
        }
    }
}
