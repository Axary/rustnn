extern crate rustnn;

fn xor_check(network: &rustnn::nn::Network) -> f32 {
    let mut total = 0.0;
    total += network.run(&[1.0, 1.0]).unwrap()[0].abs();
    total += (1.0 - network.run(&[1.0, 0.0]).unwrap()[0]).abs();
    total += (1.0 - network.run(&[0.0, 1.0]).unwrap()[0]).abs();
    total += network.run(&[0.0, 0.0]).unwrap()[0].abs();
    -total
}

fn main() {
    let generations = 10000;

    let mut environment = rustnn::Environment::new(&[2, 2, 1], 1000).unwrap();
    
    let now = std::time::Instant::now();

    for _ in 0..generations {
        environment.simple_gen(xor_check);
    }

    println!("time spend: {:?}", std::time::Instant::now().duration_since(now));
    let prev_survivor = environment.get_previous_survivor(0).unwrap();
    println!("gen {}: {}", environment.generation(), xor_check(&prev_survivor));
    println!("network.run(&[1.0, 1.0]): {:?}", prev_survivor.run(&[1.0, 1.0]).unwrap()[0]);
    println!("network.run(&[1.0, 0.0]): {:?}", prev_survivor.run(&[1.0, 0.0]).unwrap()[0]);
    println!("network.run(&[0.0, 1.0]): {:?}", prev_survivor.run(&[0.0, 1.0]).unwrap()[0]);
    println!("network.run(&[0.0, 0.0]): {:?}", prev_survivor.run(&[0.0, 0.0]).unwrap()[0]);
}