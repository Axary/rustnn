 use rand::distributions::{Distribution, Bernoulli, Uniform};

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

                values = std::iter::repeat_with(|| {
                    let section_end = offset + values.len() + 1;
                    let value = Self::calculate_value(&values, &self.weights[offset..section_end]);
                    offset = section_end;
                    value
                }).take(layer).collect();
            }

            Ok(values)
        }
    }

    pub fn test(&self, input: &[f32], expected: &[f32]) -> Result<f32, RunError> {
        let actual = self.run(input)?;

        Ok(actual.iter().zip(expected).fold(0.0, |n, (a, e)| n + (a- e).abs()))
    }

    pub fn breed(father: &Self, mother: &Self, p: f64) -> Self {
        let mut child = father.clone();

        let d = Bernoulli::new(p);
        for (c_weight, &p_weight) in child.weights.iter_mut().zip(&mother.weights) {
            if d.sample(&mut rand::thread_rng()) {
                *c_weight = p_weight;
            }
        }  

        child
    }

    pub fn mutate(&mut self, p: f64) {
        use rand::distributions::{Distribution, Bernoulli, Uniform};

        let d = Bernoulli::new(p);
        for weight in &mut self.weights {
            if d.sample(&mut rand::thread_rng()) {
                *weight = Uniform::new_inclusive(-1.0, 1.0).sample(&mut rand::thread_rng());
            }
        }  
    }
}

