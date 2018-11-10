use rayon::prelude::*;

use crate::error::*;
use crate::{Network};

use rand::Rng;
use rand::distributions::{Distribution, Bernoulli, Uniform};

pub struct Environment {
    networks: Vec<Network>,
    survivor_count: usize,
    generation: u32
}

impl Environment {
    pub fn new(network_type: &[usize], network_count: usize) -> Result<Self, CreationError> {
        Network::layers_valid(network_type)?;

        let networks = std::iter::repeat_with(|| Network::new(network_type.to_vec()).unwrap()).take(network_count).collect();

        Ok(Self {
            networks: networks,
            
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

    pub fn survivor_count(&self) -> usize {
        self.survivor_count
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }
 
    /// the return value is nonsense in case n is smaller than 2
    fn generate_survivor_distribution(n: usize) -> Vec<usize> {
        let mut rng = crate::get_rng();

        let mut survivors = vec![0, 1];
        while let Some(&last) = survivors.last() {
            let next = last + 1 + (rng.gen::<usize>() % last+1);
            
            if next < n {
                survivors.push(next);
            }
            else {
                break;
            }
        }

        survivors
    }
    
    pub fn simple_gen(&mut self, fitness_function: fn(&Network) -> f32) {
        self.generation += 1;

        let ordered_nn = Self::simple_rate(std::mem::replace(&mut self.networks, Vec::with_capacity(0)), fitness_function);
        
        let (survivor_count, networks) = Self::evolve_rated_networks(ordered_nn);

        self.survivor_count = survivor_count;
        self.networks = networks;
    }

    pub fn pair_gen(&mut self, fitness_function: fn(&Network, &Network) -> (f32, f32)) {
        self.generation += 1;

        let ordered_nn = Self::pair_rate(std::mem::replace(&mut self.networks, Vec::with_capacity(0)), fitness_function);

        let (survivor_count, networks) = Self::evolve_rated_networks(ordered_nn);

        self.survivor_count = survivor_count;
        self.networks = networks;
    }

    fn sort_fitness_vec(mut nn: Vec<(f32, Network)>) -> Vec<(f32, Network)> {
        nn.sort_unstable_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap());
        nn
    }

    pub fn simple_rate(nn: Vec<Network>, fitness_function: fn(&Network) -> f32) -> Vec<(f32, Network)> {
        Self::sort_fitness_vec(nn.into_par_iter().map(|nn| (fitness_function(&nn), nn)).collect())
    }

    pub fn pair_rate(networks: Vec<Network>, fitness_function: fn(&Network, &Network) -> (f32, f32)) -> Vec<(f32, Network)> {
        let mut nn_iter = networks.iter().enumerate();
        let mut fitness_vec: Vec<f32> = std::iter::repeat(0.0).take(networks.len()).collect();
        while let Some((a, nn_a)) = nn_iter.next() {
            for (b, nn_b) in nn_iter.clone() {
                let (fit_a, fit_b) = fitness_function(nn_a, nn_b);
                fitness_vec[a] += fit_a;
                fitness_vec[b] += fit_b;
            }
        }

        Self::sort_fitness_vec(fitness_vec.into_iter().zip(networks.into_iter()).collect())  
    }

    pub fn evolve_rated_networks(mut ordered_nn: Vec<(f32, Network)>) -> (usize, Vec<Network>) {
        let nn_count = ordered_nn.len();

        let mut networks = Vec::with_capacity(nn_count);
        let survivor_list = Self::generate_survivor_distribution(nn_count);
        let survivor_count = survivor_list.len();
        for i in survivor_list.into_iter().rev() {
            networks.push(ordered_nn.swap_remove(i).1);
        }

        // chance of breeding instead of mutation
        let b_to_m = Bernoulli::new(0.3);
        // how many weights chance during mutation
        let mutation_ratio = 0.2;
        let possible_parents = Uniform::new(0, networks.len());

        networks.append(&mut rayon::iter::repeatn((), nn_count - networks.len()).map(|_| {
            let rng = &mut crate::get_rng();
            if b_to_m.sample(rng) {
                let father = &networks[possible_parents.sample(rng)];
                let mut mother = &networks[possible_parents.sample(rng)];
                while father as *const _ == mother as *const _ {
                    mother = &networks[possible_parents.sample(rng)];
                }

                Network::breed(father, mother, 0.5)
            }
            else {
                Network::mutate(&networks[possible_parents.sample(rng)], mutation_ratio)
            }
        }).collect());

        (survivor_count, networks)
    }
}

pub trait GeneticNetwork {
    fn breed(father: &Self, mother: &Self, p: f64) -> Self;

    fn mutate(parent: &Self, p: f64) -> Self;
}

impl GeneticNetwork for Network {
    fn breed(father: &Self, mother: &Self, p: f64) -> Self {
        let layers = father.layers.clone();

        let rng = &mut crate::get_rng();
        let d = Bernoulli::new(p);
        let weights = father.weights.iter().zip(mother.weights.iter()).map(|(&f_weight, &m_weight)| if d.sample(rng) {
                f_weight
            } else {
                m_weight
            }).collect::<Vec<_>>().into_boxed_slice();

        Self {
            layers,
            weights
        }
    }

    fn mutate(parent: &Self, p: f64) -> Self {
        let layers = parent.layers.clone();
        
        let rng = &mut crate::get_rng();
        let d = Bernoulli::new(p);
        let weights = parent.weights.iter().map(|&p_weight| if d.sample(rng) {
                Uniform::new_inclusive(-1.0, 1.0).sample(rng)
            } else {
                p_weight
            }).collect::<Vec<_>>().into_boxed_slice();

        Self {
            layers,
            weights
        } 
    }
}