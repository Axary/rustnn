use rayon::prelude::*;

use crate::Network;

use rand::distributions::{Bernoulli, Distribution, Uniform};
use rand::seq::SliceRandom;
use rand::Rng;

pub struct Genetic {
    networks: Vec<Network>,
    generation: u32,
}

impl Genetic {
    pub fn new(network_type: &[usize], network_count: usize) -> Self {
        let networks = Network::multiple(network_type, network_count);

        Self {
            networks,
            generation: 0,
        }
    }

    pub fn get_network(&self, index: usize) -> Option<&Network> {
        self.networks.get(index)
    }

    pub fn get_best(&self) -> &Network {
        &self.networks[0]
    }
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// the return value is nonsense in case n is smaller than 2
    fn generate_survivor_distribution(n: usize) -> Vec<usize> {
        let mut rng = crate::get_rng();

        let mut survivors = vec![0, 1];
        while let Some(&last) = survivors.last() {
            let next = last + 1 + (rng.gen::<usize>() % (last + 1));

            if next < n {
                survivors.push(next);
            } else {
                break;
            }
        }

        survivors
    }

    pub fn simple_gen(&mut self, loss_function: fn(&Network) -> f32) {
        self.generation += 1;

        let mut indexes = self
            .networks
            .par_iter()
            .enumerate()
            .map(|(i, nn)| (loss_function(nn), i))
            .collect::<Vec<_>>();
        indexes.sort_unstable_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap());
        let indexes = indexes.into_iter().map(|(_, i)| i).collect::<Vec<_>>();

        let survivor_list = Self::generate_survivor_distribution(self.networks.len());

        let survivor_len = survivor_list.len();
        for (i, index) in survivor_list
            .into_iter()
            .map(|survivor| indexes[survivor])
            .enumerate()
        {
            if index > i {
                self.networks.swap(i, index);
            } else {
                self.networks.swap(i, indexes[i]);
            }
        }

        self.evolve_rated_networks(survivor_len);
    }

    pub fn evolve_rated_networks(&mut self, survivors: usize) {
        // chance of breeding instead of mutation
        let b_to_m = Bernoulli::new(0.3).unwrap();
        // how many weights chance during mutation
        let mutation_ratio = 0.2;

        let (alive, dead) = self.networks.split_at_mut(survivors);

        dead.par_iter_mut().for_each(|nn| {
            let rng = &mut crate::get_rng();
            if b_to_m.sample(rng) {
                let mut parents = alive.choose_multiple(rng, 2);

                let father = parents.next().unwrap();
                let mother = parents.next().unwrap();
                *nn = Network::breed(father, mother, 0.5);
            } else {
                *nn = Network::mutate(alive.choose(rng).unwrap(), mutation_ratio)
            }
        });
    }
}

pub trait GeneticNetwork {
    /// breeds a new neural network overwriting self
    fn breed(father: &Self, mother: &Self, p: f64) -> Self;

    /// mutates a new neural network overwriting self
    fn mutate(parent: &Self, p: f64) -> Self;
}

impl GeneticNetwork for Network {
    fn breed(father: &Self, mother: &Self, p: f64) -> Self {
        let layers = father.layers.clone();

        let rng = &mut crate::get_rng();
        let d = Bernoulli::new(p).unwrap();
        let weights = father
            .weights
            .iter()
            .zip(mother.weights.iter())
            .map(|(&f_weight, &m_weight)| if d.sample(rng) { f_weight } else { m_weight })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self { layers, weights }
    }

    fn mutate(parent: &Self, p: f64) -> Self {
        let layers = parent.layers.clone();

        let rng = &mut crate::get_rng();
        let d = Bernoulli::new(p).unwrap();
        let weights = parent
            .weights
            .iter()
            .map(|&p_weight| {
                if d.sample(rng) {
                    Uniform::new_inclusive(-1.0, 1.0).sample(rng)
                } else {
                    p_weight
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self { layers, weights }
    }
}
