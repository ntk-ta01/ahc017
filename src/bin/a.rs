#![allow(clippy::uninlined_format_args)]

use rand::prelude::*;
use std::collections::{BinaryHeap, HashSet, VecDeque};

const INF: i64 = 1000000000;
const TIMELIMIT: f64 = 5.45;

type Output = Vec<usize>;

fn main() {
    let timer = Timer::new();
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    let input = parse_input();
    let mut out = greedy(&input, &mut rng);
    local_search(&input, &mut out, &mut rng, &timer);
    for &o in &out {
        print!("{} ", o);
    }
    // let (score, ret, _) = compute_score(&input, &out);
    // eprintln!("score: {score}");
    // eprintln!("{ret}");
}

fn local_search<T: Rng>(input: &Input, out: &mut Output, rng: &mut T, timer: &Timer) {
    let mut vs = vec![];
    for _ in 0..25 {
        let u = rng.gen_range(0, input.ps.len());
        let mut v = rng.gen_range(0, input.ps.len());
        while u == v {
            v = rng.gen_range(0, input.ps.len());
        }
        vs.push((u, v));
    }
    let mut graphs = {
        let mut gs = vec![vec![]];
        for day in 1..=input.d {
            gs.push(get_graph(input, out, day));
        }
        gs
    };
    let mut paths = {
        let mut paths = vec![0];
        #[allow(clippy::needless_range_loop)]
        for day in 1..=input.d {
            let mut s = 0;
            for &(u, v) in vs.iter() {
                s += compute_path(&graphs[day], u, v);
            }
            paths.push(s);
        }
        paths
    };
    let mut counts = vec![0; input.d + 1];
    for i in 0..input.es.len() {
        counts[out[i]] += 1;
    }
    let mut c = 0;
    loop {
        c += 1;
        if c > 100 {
            if rng.gen_bool(0.01) {
                let mut vs2 = vec![];
                for _ in 0..25 {
                    let u = rng.gen_range(0, input.ps.len());
                    let mut v = rng.gen_range(0, input.ps.len());
                    while u == v {
                        v = rng.gen_range(0, input.ps.len());
                    }
                    vs2.push((u, v));
                }
                vs = vs2;
            }
            if TIMELIMIT < timer.get_time() {
                break;
            }
            c = 0;
        }
        // 工事する辺をmoveする近傍
        // move先の日を決める
        let to_days = (1..=input.d)
            .filter(|&d| counts[d] < input.k)
            .collect::<Vec<_>>();
        if to_days.is_empty() {
            break;
        }
        let day_to = to_days[rng.gen_range(0, to_days.len())];
        // move先の日の橋を列挙する
        let mut is_bridge = vec![false; input.es.len()];
        {
            let lowlink = LowLink::new(std::mem::take(&mut graphs[day_to]));
            for (i, e) in input.es.iter().enumerate() {
                if lowlink.bridges.contains(&(e.0, e.1)) {
                    is_bridge[i] = true;
                }
            }
        }
        // move元の工事と、工事が行われる日を特定する
        let day_from = rng.gen_range(1, input.d + 1);
        let repairs = out
            .iter()
            .enumerate()
            .filter(|(i, o)| **o == day_from && !is_bridge[*i])
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        if repairs.is_empty() {
            continue;
        }
        let e = repairs[rng.gen_range(0, repairs.len())];
        out[e] = day_to;
        // moveしたときの評価をする
        let mut new_score_from = 0;
        let graph_day_from = get_graph(input, out, day_from);
        for &(u, v) in vs.iter() {
            new_score_from += compute_path(&graph_day_from, u, v);
        }
        let mut new_score_to = 0;
        let graph_day_to = get_graph(input, out, day_to);
        for &(u, v) in vs.iter() {
            new_score_to += compute_path(&graph_day_to, u, v);
        }

        let score = paths[day_from] + paths[day_to];
        if score > new_score_from + new_score_to {
            graphs[day_from] = graph_day_from;
            graphs[day_to] = graph_day_to;
            paths[day_from] = new_score_from;
            paths[day_to] = new_score_to;
            counts[day_from] -= 1;
            counts[day_to] += 1;
        } else {
            out[e] = day_from;
            graphs[day_to] = get_graph(input, out, day_to);
        }
    }
    // eprintln!("{c}");
}

fn greedy<T: Rng>(input: &Input, rng: &mut T) -> Output {
    let mut out = vec![0; input.es.len()];
    let mut day = 1;
    let mut counts = vec![0; input.d + 1];
    let mut is_repaired_today = vec![false; input.es.len()];
    let mut is_bridge = vec![false; input.es.len()];
    let mut edges = (0..input.es.len()).collect::<Vec<_>>();
    {
        let g = get_graph(input, &out, 1);
        edges.sort_by_key(|&i| std::cmp::min(g[input.es[i].0].len(), g[input.es[i].1].len()));
    }
    let mut edges = edges.into_iter().collect::<VecDeque<_>>();
    for _ in 0..input.es.len() {
        if input.d < day {
            break;
        }
        // 工事する辺を決定
        if edges.iter().all(|&e| is_bridge[e]) {
            day += 1;
            is_repaired_today = vec![false; input.es.len()];
            is_bridge = vec![false; input.es.len()];
            continue;
        }
        let mut edge = edges.pop_front().unwrap();
        while is_bridge[edge] {
            edges.push_back(edge);
            edge = edges.pop_front().unwrap();
        }
        is_repaired_today[edge] = true;
        out[edge] = day;
        counts[day] += 1;
        // 工事する辺が閾値を超えたらdayを進める
        let upper = (input.es.len() + input.d - 1) / input.d;
        if (upper).min(input.k) <= counts[day] {
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
    #[allow(clippy::needless_range_loop)]
    for day in 1..=input.d {
        if input.k <= counts[day] {
            continue;
        }
        let mut is_bridge = vec![false; input.es.len()];
        let mut graph = get_graph(input, &out, day);
        let mut lowlink = LowLink::new(graph);
        for (i, e) in input.es.iter().enumerate() {
            if lowlink.bridges.contains(&(e.0, e.1)) {
                is_bridge[i] = true;
            }
        }
        while edges.iter().any(|&e| !is_bridge[e]) {
            let mut edge = edges.pop_front().unwrap();
            while is_bridge[edge] {
                edges.push_back(edge);
                edge = edges.pop_front().unwrap();
            }
            out[edge] = day;
            counts[day] += 1;
            if input.k <= counts[day] {
                break;
            }
            is_bridge = vec![false; input.es.len()];
            graph = get_graph(input, &out, day);
            lowlink = LowLink::new(graph);
            for (i, e) in input.es.iter().enumerate() {
                if lowlink.bridges.contains(&(e.0, e.1)) {
                    is_bridge[i] = true;
                }
            }
        }
    }
    let mut days = (1..=input.d)
        .filter(|&d| counts[d] < input.k)
        .collect::<Vec<_>>();
    for o in out.iter_mut() {
        if *o == 0 {
            // *o = rng.gen_range(1, input.d + 1);
            let d = rng.gen_range(0, days.len());
            *o = days[d];
            counts[days[d]] += 1;
            if counts[days[d]] >= input.k {
                days.remove(d);
            }
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

fn compute_path(g: &Vec<Vec<(usize, i64)>>, s: usize, t: usize) -> i64 {
    // let mut prev = vec![!0, g.len()];
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
                // prev[v] = u;
                que.push((-d2, v));
            }
        }
    }
    // let mut path = vec![];
    // let mut cur = t;
    // while cur != 0 {
    //     path.push(cur);
    //     cur = prev[cur];
    // }
    // path.reverse();
    dist[t]
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
    let mut num = 0;
    let dist0 = compute_dist_matrix(input, out, !0);
    let mut fs = vec![];
    // let mut unreachable = false; // ntk add
    for day in 1..=input.d {
        let dist = compute_dist_matrix(input, out, day);
        let mut tmp = 0;
        for i in 0..input.ps.len() {
            for j in i + 1..input.ps.len() {
                // ntk add
                // if dist[i][j] == INF {
                //     unreachable = true;
                // }
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
    // let ret = if unreachable {
    //     String::from("unreachble")
    // } else {
    //     String::new()
    // };
    // (avg.round() as i64, ret, fs)
    // ntk add
    (avg.round() as i64, String::new(), fs)
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
