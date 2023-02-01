// #![allow(clippy::uninlined_format_args)]

use rand::prelude::*;
use std::collections::{BinaryHeap, HashSet, VecDeque};

const INF: i64 = 1000000000;
// const TIMELIMIT: f64 = 6.0;

type Output = Vec<usize>;

fn main() {
    // let timer = Timer::new();
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    let input = parse_input();
    let out = greedy(&input, &mut rng);
    for &o in &out {
        print!("{o} ");
    }
    // let (score, ret, _) = compute_score(&input, &out);
    // eprintln!("score: {score}");
}

fn greedy<T: Rng>(input: &Input, rng: &mut T) -> Output {
    let mut out = vec![0; input.es.len()];
    let mut day = 1;
    let mut count = 0;
    let mut is_selected = vec![false; input.es.len()];
    let mut is_repaired_today = vec![false; input.es.len()];
    let mut is_bridge = vec![false; input.es.len()];
    let mut edges = (0..input.es.len()).collect::<VecDeque<_>>();
    for _ in 0..input.es.len() {
        if input.d < day {
            break;
        }
        // 工事する辺を決定
        let mut edge = edges.pop_front().unwrap();
        if edges.iter().all(|&e| is_selected[e] || is_bridge[e]) {
            edges.push_back(edge);
            count = 0;
            day += 1;
            is_repaired_today = vec![false; input.es.len()];
            is_bridge = vec![false; input.es.len()];
            continue;
        }
        while is_selected[edge] || is_bridge[edge] {
            edges.push_back(edge);
            edge = edges.pop_front().unwrap();
        }
        is_selected[edge] = true;
        is_repaired_today[edge] = true;
        out[edge] = day;
        count += 1;
        // 工事する辺が閾値を超えたらdayを進める
        if (input.es.len() + input.d - 1) / input.d <= count {
            count = 0;
            day += 1;
            is_repaired_today = vec![false; input.es.len()];
            is_bridge = vec![false; input.es.len()];
        }
        // 橋を計算
        {
            let graph = get_graph(input, &out, day);
            let lowlink = LowLink::new(graph);
            for (i, e) in input.es.iter().enumerate() {
                if lowlink.bridges.contains(&(e.0, e.1)) {
                    is_bridge[i] = true;
                }
            }
        }
    }
    for o in out.iter_mut() {
        if *o == 0 {
            *o = rng.gen_range(1, input.d + 1);
        }
    }
    out
}

#[allow(dead_code)]
fn get_graph(input: &Input, out: &Output, day: usize) -> Vec<Vec<(usize, i64)>> {
    let mut g = vec![vec![]; input.ps.len()];
    for (e, &o) in out.iter().enumerate().take(input.es.len()) {
        if o != day {
            let (u, v, w) = input.es[e];
            g[u].push((v, w));
            g[v].push((u, w));
        }
    }
    g
}

#[allow(dead_code)]
fn compute_dist(g: &Vec<Vec<(usize, i64)>>, s: usize) -> Vec<i64> {
    let mut dist = vec![INF; g.len()];
    let mut que = BinaryHeap::new();
    que.push((0, s));
    dist[s] = 0;
    while let Some((d, u)) = que.pop() {
        let d = -d;
        if dist[u] != d {
            continue;
        }
        for &(v, w) in &g[u] {
            let d2 = d + w;
            if dist[v] > d2 {
                dist[v] = d2;
                que.push((-d2, v));
            }
        }
    }
    dist
}

#[allow(dead_code)]
fn compute_dist_matrix(input: &Input, out: &Output, day: usize) -> Vec<Vec<i64>> {
    let g = get_graph(input, out, day);
    let mut dist = vec![];
    for s in 0..input.ps.len() {
        dist.push(compute_dist(&g, s));
    }
    dist
}

#[allow(dead_code)]
fn compute_score(input: &Input, out: &Output) -> (i64, String, Vec<f64>) {
    let mut count = vec![0; input.d + 1];
    for i in 0..input.es.len() {
        count[out[i]] += 1;
    }
    for (i, &ci) in count.iter().enumerate().skip(1) {
        if ci > input.k {
            return (
                0,
                format!(
                    "The number of edges to be repaired on day {} has exceeded the limit. ({} > {})",
                    i, ci, input.k
                ),
                vec![],
            );
        }
    }
    let mut num = 0;
    let dist0 = compute_dist_matrix(input, out, !0);
    let mut fs = vec![];
    let mut unreachable = false; // ntk add
    for day in 1..=input.d {
        let dist = compute_dist_matrix(input, out, day);
        let mut tmp = 0;
        for i in 0..input.ps.len() {
            for j in i + 1..input.ps.len() {
                // ntk add
                if dist[i][j] == INF {
                    unreachable = true;
                }
                // ntk add
                tmp += dist[i][j] - dist0[i][j];
            }
        }
        num += tmp;
        fs.push(tmp as f64 / (input.ps.len() * (input.ps.len() - 1) / 2) as f64);
    }
    let den = input.d * input.ps.len() * (input.ps.len() - 1) / 2;
    let avg = num as f64 / den as f64 * 1000.0;
    // ntk add
    let ret = if unreachable {
        String::from("unreachble")
    } else {
        String::new()
    };
    (avg.round() as i64, ret, fs)
    // ntk add
    // (avg.round() as i64, String::new(), fs)
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

struct LowLink {
    graph: Vec<Vec<(usize, i64)>>,
    used: Vec<bool>,
    ord: Vec<usize>,
    low: Vec<usize>,
    bridges: HashSet<(usize, usize)>,
}

impl LowLink {
    fn new(g: Vec<Vec<(usize, i64)>>) -> Self {
        let used = vec![false; g.len()];
        let ord = vec![0; g.len()];
        let low = vec![0; g.len()];
        let bridges = HashSet::new();
        let mut lowlink = LowLink {
            graph: g,
            used,
            ord,
            low,
            bridges,
        };
        let mut k = 0;
        for i in 0..lowlink.graph.len() {
            if !lowlink.used[i] {
                k = lowlink.dfs(i, k, !0);
            }
        }
        lowlink
    }

    fn dfs(&mut self, id: usize, mut k: usize, par: usize) -> usize {
        self.used[id] = true;
        self.ord[id] = k;
        k += 1;
        self.low[id] = self.ord[id];
        for i in 0..self.graph[id].len() {
            let e = self.graph[id][i];
            if !self.used[e.0] {
                k = self.dfs(e.0, k, id);
                self.low[id] = std::cmp::min(self.low[id], self.low[e.0]);
                if self.ord[id] < self.low[e.0] {
                    self.bridges
                        .insert((std::cmp::min(id, e.0), std::cmp::max(id, e.0)));
                }
            } else if e.0 != par {
                self.low[id] = std::cmp::min(self.low[id], self.ord[e.0]);
            }
        }
        k
    }
}

#[allow(dead_code)]
fn get_time() -> f64 {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    t.as_secs() as f64 + t.subsec_nanos() as f64 * 1e-9
}

struct Timer {
    start_time: f64,
}

#[allow(dead_code)]
impl Timer {
    fn new() -> Timer {
        Timer {
            start_time: get_time(),
        }
    }

    fn get_time(&self) -> f64 {
        get_time() - self.start_time
    }
}
