use rand::distributions::{Distribution, Bernoulli, Uniform};

use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Network {
    layers: Vec<usize>,
    weights: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CreationError {
    NotEnoughLayers,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunError {
    WrongInputCount,
}

impl Network {
    pub fn is_valid_type(layers: &[usize]) -> Result<(), CreationError> {
        if layers.len() < 2 || layers.contains(&0) {
            Err(CreationError::NotEnoughLayers)
        }
        else {
            Ok(())
        }
    }

    pub fn new(layers:  Vec<usize>) -> Result<Self, CreationError> {
        Self::is_valid_type(&layers)?;

        let (weight_count, _) = layers.iter().fold((0, 0), |(total, node_c), &layer| (total + node_c * layer, layer + 1));

        Ok(Network {
            layers: layers,
            weights: std::iter::repeat_with(|| Uniform::new_inclusive(-1.0, 1.0).sample(&mut rand::thread_rng())).take(weight_count).collect()
        })
    }

    fn calculate_value(values: &[f32], weights: &[f32]) -> f32 {
        debug_assert_eq!(values.len(), weights.len() - 1);

        let mut total = 0.0;
        for i in 0..values.len() {
            total += values[i] * weights[i];
        }

        total += weights.last().unwrap();

        crate::bipolar_sigmoid(total, 10.0)
    }

    pub fn run(&self, input: &[f32]) -> Result<Vec<f32>, RunError> {
        if input.len() != self.layers[0] {
            Err(RunError::WrongInputCount)
        }
        else {
            let mut offset = 0;
            let mut values = input.to_vec();

            for &layer in self.layers.iter().skip(1) {
                let n_weights = values.len() + 1;

                values = (0..layer).into_iter().map(|i| {
                    let section_start = offset + i * n_weights;
                    let section_end = section_start + n_weights;
                    Self::calculate_value(&values, &self.weights[section_start..section_end])
                }).collect();

                offset += layer * (n_weights);
            }

            Ok(values)
        }
    }

    pub fn test(&self, input: &[f32], expected: &[f32]) -> Result<f32, RunError> {
        let actual = self.run(input)?;

        Ok(actual.iter().zip(expected).fold(0.0, |n, (a, e)| n + (a- e).abs()))
    }

    pub fn breed(father: &Self, mother: &Self, p: f64) -> Self {
        let layers = father.layers.clone();

        let d = Bernoulli::new(p);
        let weights = father.weights.iter().zip(&mother.weights).map(|(&f_weight, &m_weight)| if d.sample(&mut rand::thread_rng()) {
                f_weight
            } else {
                m_weight
            }).collect();

        Self {
            layers,
            weights
        }
    }

    pub fn mutate(parent: &Self, p: f64) -> Self {
        let layers = parent.layers.clone();
        
        let d = Bernoulli::new(p);
        let weights = parent.weights.iter().map(|&p_weight| if d.sample(&mut rand::thread_rng()) {
                Uniform::new_inclusive(-1.0, 1.0).sample(&mut rand::thread_rng())
            } else {
                p_weight
            }).collect();

        Self {
            layers,
            weights
        } 
    }
}

