// #![allow(clippy::uninlined_format_args)]

// const INF: i64 = 1000000000;
// const TIMELIMIT: f64 = 6.0;

type Output = Vec<usize>;

fn main() {
    let input = parse_input();
    let mut out: Output = vec![];
    'lp: for d in 0..input.d {
        for _ in 0..input.k {
            out.push(d);
            if out.len() == input.es.len() {
                break 'lp;
            }
        }
    }
    for o in out {
        print!("{} ", o + 1);
    }
}

#[derive(Clone, Debug)]
struct Input {
    d: usize,
    k: usize,
    ps: Vec<(i64, i64)>,
    es: Vec<(usize, usize, i64)>,
}

fn parse_input() -> Input {
    use proconio::{input, marker::Usize1};
    input! {
        n: usize, m: usize, d: usize,k: usize,
        es: [(Usize1, Usize1, i64); m],
        ps: [(i64, i64); n],
    }
    Input { d, k, ps, es }
}
