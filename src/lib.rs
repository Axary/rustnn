//! a neural network implemented in rust

pub mod env;
pub mod func;

use rand::distributions::{Distribution, Uniform};

#[derive(Clone, Debug)]
pub struct Network {
    layers: Box<[usize]>,
    weights: Box<[f32]>,
}

impl Network {
    pub fn layers_valid(layers: &[usize]) -> bool {
        !(layers.len() < 2 || layers.contains(&0))
    }

    pub fn new(layers: &[usize]) -> Self {
        assert!(Self::layers_valid(layers));

        let rng = &mut crate::get_rng();

        let (weight_count, _) = layers.iter().fold((0, 0), |(total, node_c), &layer| {
            (total + node_c * layer, layer + 1)
        });

        Self {
            layers: layers.to_owned().into_boxed_slice(),
            weights: std::iter::repeat_with(|| Uniform::new_inclusive(-1.0, 1.0).sample(rng))
                .take(weight_count)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    pub fn multiple(layers: &[usize], count: usize) -> Vec<Self> {
        std::iter::repeat_with(|| Network::new(layers))
            .take(count)
            .collect()
    }

    fn calculate_value(
        values: &[f32],
        weights: &[f32],
        activation_function: fn(f32) -> f32,
    ) -> f32 {
        debug_assert_eq!(values.len(), weights.len() - 1);

        let mut total = 0.0;
        for i in 0..values.len() {
            total += values[i] * weights[i];
        }

        total += weights.last().unwrap();

        activation_function(total)
    }

    pub fn weight(&self, layer: usize, neuron: usize, input: usize) -> f32 {
        let mut offset = 0;
        for i in 1..layer {
            offset += self.layers[i] * (self.layers[i - 1] + 1);
        }

        self.weights[offset + neuron * (self.layers[layer - 1] + 1) + input]
    }

    pub fn set_weight(&mut self, layer: usize, neuron: usize, input: usize, value: f32) {
        let mut offset = 0;
        for i in 1..layer {
            offset += self.layers[i] * (self.layers[i - 1] + 1);
        }

        self.weights[offset + neuron * (self.layers[layer - 1] + 1) + input] = value;
    }

    pub fn run_remembering_state(&self, input: &[f32]) -> Vec<Vec<f32>> {
        self.run_with_activation_function_remembering_state(input, |t| {
            crate::func::bipolar_sigmoid(t, 10.0)
        })
    }

    pub fn run_with_activation_function_remembering_state(
        &self,
        input: &[f32],
        activation_function: fn(f32) -> f32,
    ) -> Vec<Vec<f32>> {
        assert_eq!(input.len(), self.layers[0]);

        let mut offset = 0;
        let mut values = vec![input.to_vec()];

        for &neuron_count in self.layers.iter().skip(1) {
            let n_weights = values.len() + 1;

            values.push(
                (0..neuron_count)
                    .into_iter()
                    .map(|i| {
                        let section_start = offset + i * n_weights;
                        let section_end = section_start + n_weights;
                        Self::calculate_value(
                            &values.last().unwrap(),
                            &self.weights[section_start..section_end],
                            activation_function,
                        )
                    })
                    .collect(),
            );

            offset += neuron_count * (n_weights);
        }

        values
    }

    pub fn run(&self, input: &[f32]) -> Vec<f32> {
        self.run_with_activation_function(input, |t| crate::func::binary_sigmoid(t, 1.0))
    }

    pub fn run_with_activation_function(
        &self,
        input: &[f32],
        activation_function: fn(f32) -> f32,
    ) -> Vec<f32> {
        assert_eq!(input.len(), self.layers[0]);

        let mut offset = 0;
        let mut values = input.to_vec();

        for &neuron_count in self.layers.iter().skip(1) {
            let n_weights = values.len() + 1;

            values = (0..neuron_count)
                .into_iter()
                .map(|i| {
                    let section_start = offset + i * n_weights;
                    let section_end = section_start + n_weights;
                    Self::calculate_value(
                        &values,
                        &self.weights[section_start..section_end],
                        activation_function,
                    )
                })
                .collect();

            offset += neuron_count * (n_weights);
        }

        values
    }
}

#[inline(always)]
pub fn get_rng() -> impl rand::Rng {
    //rand::XorShiftRng::from_entropy()
    rand::thread_rng()
}
