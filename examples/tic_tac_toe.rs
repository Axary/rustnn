extern crate rustnn;

use rustnn::nn::Network;

#[derive(Debug, Clone, Default)]
struct Area {
    fields: [f32; 9]
}

#[derive(Copy, Clone, Debug)]
enum MoveResult {
    Ok,
    AlreadyOccupied,
    Win,
    NoTilesLeft
}

impl Area {
    fn show(&self) {
        print!("{:+}|{:+}|{:+}\n-----\n{:+}|{:+}|{:+}\n-----\n{:+}|{:+}|{:+}\n\n", self.fields[0], self.fields[1], self.fields[2], self.fields[3], self.fields[4]
        , self.fields[5], self.fields[6], self.fields[7], self.fields[8]);
    }

    fn set_stone(&mut self, player: f32, pos: usize) -> MoveResult {
        if self.fields[pos] == 0.0 {
            self.fields[pos] = player;

            let rows = [[0, 1 ,2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];

            for row in rows.iter() {
                let mut total = 0.0;
                for &field in row.iter() {
                    total += self.fields[field] * player;
                }

                if total > 2.5 {
                    return MoveResult::Win
                }
            }

            MoveResult::Ok
        }
        else {
            MoveResult::AlreadyOccupied
        }
    }
}

fn turn(nn: &Network, player: f32, area: &mut Area) -> MoveResult {
    let mut res: Vec<_> = nn.run(&area.fields).unwrap().into_iter().enumerate().collect();
    res.sort_unstable_by(|(_, a), (_,b)| b.partial_cmp(a).unwrap());
    for (choice, _) in res {
        match area.set_stone(player, choice) {
            MoveResult::Ok => return MoveResult::Ok,
            MoveResult::Win => return MoveResult::Win,
            MoveResult::NoTilesLeft => unreachable!(),
            MoveResult::AlreadyOccupied => continue,
        }
    }

    MoveResult::NoTilesLeft
}

fn turn_showcase(nn: &Network, player: f32, area: &mut Area) -> MoveResult {
    let mut res: Vec<_> = nn.run(&area.fields).unwrap().into_iter().enumerate().collect();
    println!("bot choices: {:?}", res);
    res.sort_unstable_by(|(_, a), (_,b)| b.partial_cmp(a).unwrap());
    for (choice, _) in res {
        match area.set_stone(player, choice) {
            MoveResult::Ok => return MoveResult::Ok,
            MoveResult::Win => return MoveResult::Win,
            MoveResult::NoTilesLeft => unreachable!(),
            MoveResult::AlreadyOccupied => continue,
        }
    }

    MoveResult::NoTilesLeft
}

fn turn_as_user(player: f32, area: &mut Area) -> MoveResult {
    loop {
        use std::io::BufRead;
        let input: usize = std::io::stdin().lock().lines().next().unwrap().unwrap().parse().unwrap();
        if input == 0 {
            return MoveResult::NoTilesLeft;
        }
        match area.set_stone(player, input - 1) {
            MoveResult::Ok => return MoveResult::Ok,
            MoveResult::Win => return MoveResult::Win,
            MoveResult::NoTilesLeft => unreachable!(),
            MoveResult::AlreadyOccupied => continue,
        }
    }
}

fn play(a: &Network, b: &Network) -> (f32, f32) {
    // a starts
    let mut area = Area::default();
    let fitness_a = loop {
        match turn(a, 1.0, &mut area) {
            MoveResult::Ok => (),
            MoveResult::AlreadyOccupied => unreachable!(),
            MoveResult::NoTilesLeft => break (0.0, 0.0),
            MoveResult::Win => break (1.0, -100.0),
        }

        match turn(b, -1.0, &mut area) {
            MoveResult::Ok => (),
            MoveResult::AlreadyOccupied => unreachable!(),
            MoveResult::NoTilesLeft => break (0.0, 0.0),
            MoveResult::Win => break (-100.0, 1.0),
        }
    };

    // b starts
    let mut area = Area::default();
    let fitness_b = loop {
        match turn(b, 1.0, &mut area) {
            MoveResult::Ok => (),
            MoveResult::AlreadyOccupied => unreachable!(),
            MoveResult::NoTilesLeft => break (0.0, 0.0),
            MoveResult::Win => break (-100.0, 1.0),
        }

        match turn(a, -1.0, &mut area) {
            MoveResult::Ok => (),
            MoveResult::AlreadyOccupied => unreachable!(),
            MoveResult::NoTilesLeft => break (0.0, 0.0),
            MoveResult::Win => break (1.0, -100.0),
        }
    };

    (fitness_a.0 + fitness_b.0, fitness_a.1 + fitness_b.1)
}

fn play_showcase(a: &Network) {
    loop {
        // a starts
        let mut area = Area::default();
        println!("bot starts");
        loop {
            let res = turn_showcase(a, 1.0, &mut area);
            println!("bot: {:?}", res);
            area.show();
            match res {
                MoveResult::Ok => (),
                MoveResult::AlreadyOccupied => unreachable!(),
                MoveResult::NoTilesLeft => break,
                MoveResult::Win => break,
            }
            
            let res = turn_as_user(-1.0, &mut area);
            println!("player: {:?}", res);
            area.show();
            match res {
                MoveResult::Ok => (),
                MoveResult::AlreadyOccupied => unreachable!(),
                MoveResult::NoTilesLeft => break,
                MoveResult::Win => break,
            }
        };

        println!("player starts");
        // a starts
        let mut area = Area::default();
        loop {
            let res = turn_as_user(1.0, &mut area);
            println!("player: {:?}", res);
            area.show();
            match res {
                MoveResult::Ok => (),
                MoveResult::AlreadyOccupied => unreachable!(),
                MoveResult::NoTilesLeft => break,
                MoveResult::Win => break,
            }

            let res = turn_showcase(a, -1.0, &mut area);
            println!("bot: {:?}", res);
            area.show();
            match res {
                MoveResult::Ok => (),
                MoveResult::AlreadyOccupied => unreachable!(),
                MoveResult::NoTilesLeft => break,
                MoveResult::Win => break,
            }
        };
    }
}

fn main() {
    let generations = 20000;

    let mut environment = rustnn::Environment::new(&[9, 18, 18, 9], 200).unwrap();
    
    let now = std::time::Instant::now();

    for _ in 0..generations {
        environment.pair_gen(play);
    }

    println!("time spend: {:?}", std::time::Instant::now().duration_since(now));
    let prev_survivor = environment.get_previous_survivor(0).unwrap();
    play_showcase(prev_survivor);
}