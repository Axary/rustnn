//! # rustnn
//! 
//! a neural network library written in rust with a primary focus on speed
pub mod nn;

use std::num::NonZeroUsize;

use self::nn::Network;

/// calculates the binary sigmoidal function of `t`, returning a value between `0.0` and `1.0`:
/// 
/// `sigmoid(t) = 1 / (1 + e ^ (-t * a))`
#[inline(always)]
pub fn sigmoid(t: f32, a: f32) -> f32 {
    1.0 / (1.0 + (-t * a).exp())
}

pub fn bipolar_sigmoid(t: f32, a: f32) -> f32 {
    2.0 / (1.0 + (-t * a).exp()) - 1.0
}

pub struct Environment {
    networks: Vec<Network>,
    fitness_function: fn(&Network) -> f32,

    survivor_count: usize,
    generation: u32
}

impl Environment {
    pub fn new(network_type: &[NonZeroUsize], network_count: usize, fitness_function: fn(&Network) -> f32) -> Result<Self, nn::CreationError> {
        Network::is_valid_type(network_type)?;

        let networks = (0..network_count).map(|_| Network::new(network_type).unwrap()).collect();

        Ok(Self {
            networks: networks,
            fitness_function: fitness_function,
            
            survivor_count: 0,
            generation: 0,
        })
    }

    pub fn get_network(&self, index: usize) -> Option<&Network> {
        self.networks.get(index)
    }

    pub fn get_previous_survivor(&self, index: usize) -> Option<&Network> {
        if self.survivor_count > index {
            self.networks.get(self.survivor_count - 1 - index)
        }
        else {
            None
        }
    }

    pub fn previous_survivor_count(&self) -> usize {
        self.survivor_count
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    // this function panics in case the condition `n > 1` is not met
    fn generate_survivor_list(n: usize) -> Vec<usize> {
        let mut survivors = vec![0, 1];
        while let Some(&last) = survivors.last() {
            let next = last + 1 + (rand::random::<usize>() % last+1);
            
            if next < n {
                survivors.push(next);
            }
            else {
                break;
            }
        }

        survivors
    }

    pub fn run_step(&mut self) {
        self.generation += 1;

        let ordered_nn = {
            let mut nn: Vec<_> = std::mem::replace(&mut self.networks, Vec::with_capacity(0)).into_iter()
            .map(|nn| ((self.fitness_function)(&nn), nn)).collect();
            nn.sort_unstable_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap());
            nn
        };
        
        let (survivor_count, networks) = Self::evolve_rated_networks(ordered_nn);

        self.survivor_count = survivor_count;
        self.networks = networks;
    }

    fn evolve_rated_networks(mut ordered_nn: Vec<(f32, Network)>) -> (usize, Vec<Network>) {
        let nn_count = ordered_nn.len();

        let mut networks = Vec::with_capacity(nn_count);
        let survivor_list = Self::generate_survivor_list(nn_count);
        let survivor_count = survivor_list.len();
        for i in survivor_list.into_iter().rev() {
            networks.push(ordered_nn.swap_remove(i).1);
        }

        use rand::distributions::{Distribution, Bernoulli, Uniform};
        // chance of breeding instead of mutation
        let m_to_b = Bernoulli::new(0.3);
        // how many weights chance during mutation
        let mutation_ratio = 0.3;
        let rng = &mut rand::thread_rng();
        let possible_parents = Uniform::new(0, networks.len());
        while networks.len() < nn_count {
            if m_to_b.sample(rng) {
                let father = &networks[possible_parents.sample(rng)];
                let mut mother = &networks[possible_parents.sample(rng)];
                while father as *const _ == mother as *const _ {
                    mother = &networks[possible_parents.sample(rng)];
                }
            }
            else {
                let mut offspring = networks[possible_parents.sample(rng)].clone();
                offspring.mutate(mutation_ratio);
                networks.push(offspring);
            }
        }

        (survivor_count, networks)
    }
}