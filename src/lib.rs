//! a neural network implemented in rust

pub mod func;
pub mod genetic;

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

    pub fn new(layers: Vec<usize>) ->Self {
        assert!(Self::layers_valid(&layers));

        let rng = &mut crate::get_rng();

        let (weight_count, _) = layers.iter().fold((0, 0), |(total, node_c), &layer| (total + node_c * layer, layer + 1));

        Self {
            layers: layers.into_boxed_slice(),
            weights: std::iter::repeat_with(|| Uniform::new_inclusive(-1.0, 1.0).sample(rng)).take(weight_count)
                .collect::<Vec<_>>().into_boxed_slice()
        }
    }

    fn calculate_value(values: &[f32], weights: &[f32], activation_function: fn(f32) -> f32) -> f32 {
        debug_assert_eq!(values.len(), weights.len() - 1);

        let mut total = 0.0;
        for i in 0..values.len() {
            total += values[i] * weights[i];
        }

        total += weights.last().unwrap();

        activation_function(total)
    }

    pub fn run(&self, input: &[f32]) -> Vec<f32>{
        self.run_with_activation_function(input, |t| crate::func::bipolar_sigmoid(t, 10.0))
        
    }

    pub fn run_with_activation_function(&self, input: &[f32], activation_function: fn(f32) -> f32) -> Vec<f32> {
        assert_eq!(input.len(), self.layers[0]);

        let mut offset = 0;
        let mut values = input.to_vec();

        for &layer in self.layers.iter().skip(1) {
            let n_weights = values.len() + 1;

            values = (0..layer).into_iter().map(|i| {
                let section_start = offset + i * n_weights;
                let section_end = section_start + n_weights;
                Self::calculate_value(&values, &self.weights[section_start..section_end], activation_function)
            }).collect();

            offset += layer * (n_weights);
        }

        values
    }
}

#[inline(always)]
pub fn get_rng() -> impl rand::Rng {
    //rand::XorShiftRng::from_entropy()
    rand::thread_rng()
}
