extern crate rustnn;

use std::num::NonZeroUsize;

fn test(network: &rustnn::nn::Network) -> f32 {
    let mut total = 0.0;
    total += network.run(&[1.0, 1.0]).unwrap()[0].abs();
    total += 1.0 - network.run(&[1.0, 0.0]).unwrap()[0].abs();
    total += 1.0 - network.run(&[0.0, 1.0]).unwrap()[0].abs();
    total += network.run(&[0.0, 0.0]).unwrap()[0].abs();
    total
}

fn main() {
    let mut environment = rustnn::Environment::new(&[NonZeroUsize::new(2).unwrap(), NonZeroUsize::new(2).unwrap(), NonZeroUsize::new(1).unwrap()],
        1000, test).unwrap();
    
    let now = std::time::Instant::now();
    loop {
        environment.run_step();
        if environment.generation() % 1000 == 0 {
            println!("time spend: {:?}", std::time::Instant::now().duration_since(now));
            let prev_survivor = environment.get_previous_survivor(0).unwrap();
            println!("gen {}: {}", environment.generation(), test(&prev_survivor));
            println!("{:?}", prev_survivor);
            println!("network.run(&[1.0, 1.0]): {:?}", prev_survivor.run(&[1.0, 1.0]).unwrap()[0]);
            println!("network.run(&[1.0, 0.0]): {:?}", prev_survivor.run(&[1.0, 0.0]).unwrap()[0]);
            println!("network.run(&[0.0, 1.0]): {:?}", prev_survivor.run(&[0.0, 1.0]).unwrap()[0]);
            println!("network.run(&[0.0, 0.0]): {:?}", prev_survivor.run(&[0.0, 0.0]).unwrap()[0]);
            
            break;
        }
        
    }
}