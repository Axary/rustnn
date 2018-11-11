# rnn (rust neural network) 

A neural network implemented in Rust.

`rnn` is currently WIP and far from stable. Once [const generics](https://github.com/rust-lang/rust/issues/44580) are available on nightly, this crate will be redesigned and about *50 %* faster.

## Example

Add this to your `Cargo.toml`:

```toml
[dependencies]
rnn = "0.1"
```

and this to your crate root if you are not using the [2018 edition](https://rust-lang-nursery.github.io/edition-guide/rust-2018/index.html) of Rust:

```rust
extern crate rnn;
```

Let's try to train a network to resemble the logical function `f(a, b, c) = (a AND b) XOR c` by using a *genetic algorithm*.

```rust
// in `fn main()`

let network_count = 250;
// 3 input nodes, 1 hidden layer with 3 nodes and one output node
let layers = [3, 3, 1];

let mut environment = rnn::env::Genetic::new(&layers, network_count);

// run 1000 generations
for _ in 0..1000 {
    environment.simple_gen(loss_function);
}
```

The [loss function](https://en.wikipedia.org/wiki/Loss_function) is used to select the networks which perform best at the desired tasks.
For this example let's check every possible input by using a truth table and return the sum of the squared error.

```rust
// outside of `fn main()`

const TRUTH_TABLE: [([f32; 3], f32); 8] =
    [
        ([0.0, 0.0, 0.0], 0.0),
        ([0.0, 0.0, 1.0], 1.0),
        ([0.0, 1.0, 0.0], 0.0),
        ([0.0, 1.0, 1.0], 1.0),
        ([1.0, 0.0, 0.0], 0.0),
        ([1.0, 0.0, 1.0], 1.0),
        ([1.0, 1.0, 0.0], 1.0),
        ([1.0, 1.0, 1.0], 0.0),
    ];

fn loss_function(network: &rnn::Network) -> f32 {    
    use rnn::func::squared_error;

    TRUTH_TABLE.iter().fold(0.0, |total, &(input, ideal)| {
        total + squared_error(&[ideal], &network.run(&input))
    })
}
```

To check if the training was effective let's use a modified version of the `loss_function` which simply prints the results.

```rust
// outside of `fn main()`
fn showcase(network: &Network) {    
    use rnn::func::squared_error;

    println!("fitness of the best survivor: {}", loss_function(network);
    println!("individual results:");

    TRUTH_TABLE.iter().for_each(|&(input, ideal)| {
        println!("{:?} => ideal: {:?}, actual: {:?}", input, ideal[0], network.run(&input));
    })
}
```

with the best survivor of the last generation:
```rust
// continue in `fn main()`
showcase(environment.get_best());
```

The full example can be found in [examples/readme.rs](examples/readme.rs)

## Todo

1. add a `Environment` to train the neural networks with [back propagation](https://en.wikipedia.org/wiki/Backpropagation).
1. add a more complex example, something something image recognition.
1. add [serde](https://crates.io/crates/serde) support to resume training between sessions.
3. improve the API


## License

`rnn` is distributed under the terms of both the MIT license and the
Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT), and
[COPYRIGHT](COPYRIGHT) for details.


