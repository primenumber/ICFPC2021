use clap::{App, Arg, ArgMatches, SubCommand};
use rand::distributions::{Distribution, Uniform};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_distr::Binomial;
use serde::{Deserialize, Serialize};
use std::cmp::min;
use std::collections::VecDeque;
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader, BufWriter};
use std::mem;

#[derive(PartialEq, Eq, Debug, Clone, Copy, Deserialize, Serialize)]
struct Point(i64, i64);

impl Point {
    fn dot(&self, rhs: Point) -> i64 {
        self.0 * rhs.0 + self.1 * rhs.1
    }

    fn norm(&self) -> i64 {
        self.dot(*self)
    }

    fn distance_sq(&self, rhs: Point) -> i64 {
        self.sub(rhs).norm()
    }

    fn cross(&self, rhs: Point) -> i64 {
        self.0 * rhs.1 - self.1 * rhs.0
    }

    fn sub(&self, rhs: Point) -> Point {
        Point(self.0 - rhs.0, self.1 - rhs.1)
    }

    fn add(&self, rhs: Point) -> Point {
        Point(self.0 + rhs.0, self.1 + rhs.1)
    }

    fn scale(&self, rhs: f64) -> Point {
        Point((self.0 as f64 * rhs) as i64, (self.1 as f64 * rhs) as i64)
    }

    fn rotate(&self, theta: f64) -> Point {
        Point(
            (self.0 as f64 * theta.cos() - self.1 as f64 * theta.sin()) as i64,
            (self.0 as f64 * theta.sin() + self.1 as f64 * theta.cos()) as i64,
        )
    }

    fn is_in_hole(&self, hole: &[Point]) -> bool {
        let mut parity = false;
        for (idx, &p) in hole.iter().enumerate() {
            let q = hole[(idx + 1) % hole.len()];
            let mut dp = p.sub(*self);
            let mut dq = q.sub(*self);
            if dp.1 > dq.1 {
                mem::swap(&mut dp, &mut dq);
            }
            if dp.1 <= 0 && 0 < dq.1 {
                if dp.cross(dq) < 0 {
                    parity = !parity;
                }
            }
            if dp.cross(dq) == 0 && dp.dot(dq) == 0 {
                return true;
            }
        }
        parity
    }
}

// counter clockwise
fn ccw(a: Point, b: Point, c: Point) -> i64 {
    let ab = b.sub(a);
    let ac = c.sub(a);
    if ab.cross(ac) > 0 {
        return 1; // counter clockwise
    } else if ab.cross(ac) < 0 {
        return -1; // clockwise
    }
    if ab.dot(ac) < 0 {
        return 2; // c--a--b on line
    }
    if ab.norm() < ac.norm() {
        return -2; // a--b--c on line
    }
    return 0; // a--c--b on line
}

#[derive(Clone, Copy)]
struct Segment(Point, Point);

impl Segment {
    fn length(&self) -> i64 {
        let d = self.0.sub(self.1);
        d.norm()
    }

    fn is_cross(&self, other: Self) -> bool {
        let x = ccw(self.0, self.1, other.0);
        let y = ccw(self.0, self.1, other.1);
        let z = ccw(other.0, other.1, self.0);
        let w = ccw(other.0, other.1, self.1);
        if x == -1 && y == 1 && z == 0 && w == -1 {
            true
        } else if x == 1 && y == -1 && z == -1 && w == 0 {
            true
        } else {
            x * y == -1 && z * w == -1
        }
    }

    fn proj(&self, p: Point) -> Point {
        let ap = p.sub(self.0);
        let ba = self.0.sub(self.1);
        let t = ap.dot(ba) as f64 / ba.norm() as f64;
        self.0.add(ba.scale(t))
    }

    fn mirror(&self, p: Point) -> Point {
        self.proj(p).scale(2.0).sub(p)
    }
}

#[derive(Serialize, Deserialize)]
struct Figure {
    vertices: Vec<Point>,
    edges: Vec<(usize, usize)>,
}

#[derive(Serialize, Deserialize)]
enum BonusType {
    GLOBALIST,
    BREAK_A_LEG,
    WALLHACK,
}

#[derive(Serialize, Deserialize)]
struct Bonus {
    bonus: BonusType,
    problem: usize,
    position: Point,
}

#[derive(Serialize, Deserialize)]
struct Problem {
    bonuses: Vec<Bonus>,
    hole: Vec<Point>,
    figure: Figure,
    epsilon: u32,
}

#[derive(Serialize, Deserialize, Clone)]
struct Pose {
    vertices: Vec<Point>,
}

impl Pose {
    fn is_in_hole(&self, edges: &[(usize, usize)], hole: &[Point]) -> bool {
        for &p in &self.vertices {
            if !p.is_in_hole(hole) {
                return false;
            }
        }
        for &(u, v) in edges {
            let s = Segment(self.vertices[u], self.vertices[v]);
            for (idx, &p) in hole.iter().enumerate() {
                let q = hole[(idx + 1) % hole.len()];
                let t = Segment(p, q);
                if s.is_cross(t) {
                    return false;
                }
                if ccw(s.0, s.1, q) == 0 && q != s.0 && q != s.1 {
                    let r = hole[(idx + 2) % hole.len()];
                    let qp = p.sub(q);
                    let qr = r.sub(q);
                    let qa = s.0.sub(q);
                    let qb = s.1.sub(q);
                    if qp.cross(qr) < 0 {
                        return false;
                    }
                    if qp.cross(qa) > 0 && qa.cross(qr) > 0 {
                        return false;
                    }
                    if qp.cross(qb) > 0 && qb.cross(qr) > 0 {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn is_in_hole_single_point(&self, prob: &Problem, index: usize) -> bool {
        if !self.vertices[index].is_in_hole(&prob.hole) {
            return false;
        }
        let n = prob.hole.len();
        for &(u, v) in &prob.figure.edges {
            if u != index && v != index {
                continue;
            }
            let s = Segment(self.vertices[u], self.vertices[v]);
            for (idx, &p) in prob.hole.iter().enumerate() {
                let q = prob.hole[(idx + 1) % n];
                let t = Segment(p, q);
                if s.is_cross(t) {
                    return false;
                }
                if ccw(s.0, s.1, q) == 0 && q != s.0 && q != s.1 {
                    let r = prob.hole[(idx + 2) % n];
                    let qp = p.sub(q);
                    let qr = r.sub(q);
                    let qa = s.0.sub(q);
                    let qb = s.1.sub(q);
                    if qp.cross(qr) < 0 {
                        return false;
                    }
                    if qp.cross(qa) > 0 && qa.cross(qr) > 0 {
                        return false;
                    }
                    if qp.cross(qb) > 0 && qb.cross(qr) > 0 {
                        return false;
                    }
                }
            }
        }
        true
    }
}

fn parse_problem(input_path: &str) -> std::io::Result<Problem> {
    let in_f = File::open(input_path)?;
    let mut reader = BufReader::new(in_f);
    let mut prob_str = String::new();
    reader.read_to_string(&mut prob_str)?;
    let prob_json: Problem = serde_json::from_str(&prob_str).unwrap();
    Ok(prob_json)
}

fn parse_pose(input_path: &str) -> std::io::Result<Pose> {
    let in_f = File::open(input_path)?;
    let mut reader = BufReader::new(in_f);
    let mut prob_str = String::new();
    reader.read_to_string(&mut prob_str)?;
    let prob_json: Pose = serde_json::from_str(&prob_str).unwrap();
    Ok(prob_json)
}

struct NearestCache {
    nearest_id: Vec<usize>,
    nearest_distance: Vec<i64>,
}

impl NearestCache {
    fn new(pose: &Pose, prob: &Problem) -> NearestCache {
        let mut nearest_id = Vec::new();
        let mut nearest_distance = Vec::new();
        for &p in &prob.hole {
            let mut min_d = 1_000_000_000_000_000_000;
            let mut min_id = 0;
            for (idx, &q) in pose.vertices.iter().enumerate() {
                let d = p.distance_sq(q);
                if d < min_d {
                    min_d = d;
                    min_id = idx;
                }
            }
            nearest_id.push(min_id);
            nearest_distance.push(min_d);
        }
        NearestCache {
            nearest_id,
            nearest_distance,
        }
    }

    fn sum(&self) -> i64 {
        let mut sum = 0;
        for d in &self.nearest_distance {
            sum += d;
        }
        sum
    }

    fn update(&mut self, pose: &Pose, prob: &Problem, updated_id: usize) {
        for (idx, &p) in prob.hole.iter().enumerate() {
            if self.nearest_id[idx] == updated_id {
                let new_distance = p.distance_sq(pose.vertices[updated_id]);
                if new_distance > self.nearest_distance[idx] {
                    let mut min_d = new_distance;
                    let mut min_id = idx;
                    for (j, &q) in pose.vertices.iter().enumerate() {
                        let d = p.distance_sq(q);
                        if d < min_d {
                            min_d = d;
                            min_id = j;
                        }
                    }
                    self.nearest_id[idx] = min_id;
                    self.nearest_distance[idx] = min_d;
                }
            } else {
                let new_distance = p.distance_sq(pose.vertices[updated_id]);
                if new_distance < self.nearest_distance[idx] {
                    self.nearest_id[idx] = updated_id;
                    self.nearest_distance[idx] = new_distance;
                }
            }
        }
    }
}

fn cost_unchecked(pose: &Pose, prob: &Problem, cache: &NearestCache, weight: f64) -> f64 {
    let mut result = 0.0;
    for &(u, v) in &prob.figure.edges {
        let orig_seg = Segment(prob.figure.vertices[u], prob.figure.vertices[v]);
        let pose_seg = Segment(pose.vertices[u], pose.vertices[v]);
        let ratio = pose_seg.length() as f64 / orig_seg.length() as f64;
        result +=
            (((ratio - 1.0).abs() * 1e6 - prob.epsilon as f64 * 0.9999) * 1e5 / weight).max(0.0);
    }
    result += cache.sum() as f64;
    result
}

fn cost(pose: &Pose, prob: &Problem, cache: &NearestCache, weight: f64) -> f64 {
    if !pose.is_in_hole(&prob.figure.edges, &prob.hole) {
        return 1e12;
    }
    cost_unchecked(pose, prob, cache, weight)
}

fn move_one(
    pose: &mut Pose,
    prob: &Problem,
    rng: &mut SmallRng,
    temp: f64,
    cache: &mut NearestCache,
) -> bool {
    let old_cost = cost_unchecked(pose, prob, cache, temp);
    let idx = Uniform::from(0..pose.vertices.len()).sample(rng);
    let p = pose.vertices[idx];
    let dx = Binomial::new(16, 0.5).unwrap().sample(rng) as i64 - 8;
    let dy = Binomial::new(16, 0.5).unwrap().sample(rng) as i64 - 8;
    let np = Point(p.0 + dx, p.1 + dy);
    pose.vertices[idx] = np;
    if !pose.is_in_hole_single_point(prob, idx) {
        pose.vertices[idx] = p;
        return false;
    }
    cache.update(pose, prob, idx);
    let new_cost = cost_unchecked(pose, prob, cache, temp);
    if new_cost <= old_cost {
        return true;
    }
    if rng.gen::<f64>() > ((old_cost - new_cost) / temp).exp() {
        pose.vertices[idx] = p;
        cache.update(pose, prob, idx);
    }
    false
}

fn flip_one(
    pose: &mut Pose,
    prob: &Problem,
    rng: &mut SmallRng,
    temp: f64,
    cache: &mut NearestCache,
) -> bool {
    let old_cost = cost_unchecked(pose, prob, cache, temp);
    let mut degrees = vec![0; pose.vertices.len()];
    for &(u, v) in &prob.figure.edges {
        degrees[u] += 1;
        degrees[v] += 1;
    }
    let mut flippables = Vec::new();
    for idx in 0..pose.vertices.len() {
        if degrees[idx] == 2 {
            flippables.push(idx);
        }
    }
    if flippables.is_empty() {
        return false;
    }
    let idx = flippables[Uniform::from(0..flippables.len()).sample(rng)];
    let mut neighbors = Vec::new();
    for &(u, v) in &prob.figure.edges {
        if u == idx {
            neighbors.push(v);
        } else if v == idx {
            neighbors.push(u);
        }
    }
    assert!(neighbors.len() == 2);

    let line = Segment(pose.vertices[neighbors[0]], pose.vertices[neighbors[1]]);
    let p = pose.vertices[idx];
    pose.vertices[idx] = line.mirror(p);

    if !pose.is_in_hole_single_point(prob, idx) {
        pose.vertices[idx] = p;
        return false;
    }

    cache.update(pose, prob, idx);
    let new_cost = cost_unchecked(pose, prob, cache, temp);
    if new_cost <= old_cost {
        return true;
    }
    if rng.gen::<f64>() > ((old_cost - new_cost) / temp).exp() {
        pose.vertices[idx] = p;
        cache.update(pose, prob, idx);
    }
    false
}

fn move_two(
    pose: &mut Pose,
    prob: &Problem,
    rng: &mut SmallRng,
    temp: f64,
    cache: &mut NearestCache,
) -> bool {
    let old_cost = cost_unchecked(pose, prob, cache, temp);
    let idx = Uniform::from(0..prob.figure.edges.len()).sample(rng);
    let (u, v) = prob.figure.edges[idx];
    let p = pose.vertices[u];
    let q = pose.vertices[v];
    let dx = Binomial::new(16, 0.5).unwrap().sample(rng) as i64 - 8;
    let dy = Binomial::new(16, 0.5).unwrap().sample(rng) as i64 - 8;
    let np = Point(p.0 + dx, p.1 + dy);
    let nq = Point(q.0 + dx, q.1 + dy);
    pose.vertices[u] = np;
    pose.vertices[v] = nq;

    if !pose.is_in_hole_single_point(prob, u) || !pose.is_in_hole_single_point(prob, v) {
        pose.vertices[u] = p;
        pose.vertices[v] = q;
        return false;
    }
    cache.update(pose, prob, u);
    cache.update(pose, prob, v);
    let new_cost = cost_unchecked(pose, prob, cache, temp);
    if new_cost <= old_cost {
        return true;
    }
    if rng.gen::<f64>() > ((old_cost - new_cost) / temp).exp() {
        pose.vertices[u] = p;
        pose.vertices[v] = q;
        cache.update(pose, prob, u);
        cache.update(pose, prob, v);
    }
    false
}

fn rotate_two(
    pose: &mut Pose,
    prob: &Problem,
    rng: &mut SmallRng,
    temp: f64,
    cache: &mut NearestCache,
) -> bool {
    let old_cost = cost_unchecked(pose, prob, cache, temp);
    let idx = Uniform::from(0..prob.figure.edges.len()).sample(rng);
    let (u, v) = prob.figure.edges[idx];
    let p = pose.vertices[u];
    let q = pose.vertices[v];
    let rad = rng.gen::<f64>() - 0.5;
    let g = p.add(q).scale(0.5);
    let np = p.sub(g).rotate(rad).add(g);
    let nq = q.sub(g).rotate(rad).add(g);
    pose.vertices[u] = np;
    pose.vertices[v] = nq;

    if !pose.is_in_hole_single_point(prob, u) || !pose.is_in_hole_single_point(prob, v) {
        pose.vertices[u] = p;
        pose.vertices[v] = q;
        return false;
    }
    cache.update(pose, prob, u);
    cache.update(pose, prob, v);
    let new_cost = cost_unchecked(pose, prob, cache, temp);
    if new_cost <= old_cost {
        return true;
    }
    if rng.gen::<f64>() > ((old_cost - new_cost) / temp).exp() {
        pose.vertices[u] = p;
        pose.vertices[v] = q;
        cache.update(pose, prob, u);
        cache.update(pose, prob, v);
    }
    false
}

fn move_several(
    pose: &mut Pose,
    prob: &Problem,
    rng: &mut SmallRng,
    temp: f64,
    cache: &mut NearestCache,
) -> bool {
    let old_cost = cost_unchecked(pose, prob, cache, temp);
    let size = Uniform::from(2..=pose.vertices.len()).sample(rng);
    let sink = Uniform::from(0..pose.vertices.len()).sample(rng);
    let mut neighbors = vec![Vec::new(); pose.vertices.len()];
    for &(u, v) in &prob.figure.edges {
        neighbors[u].push(v);
        neighbors[v].push(u);
    }
    let mut visited = vec![false; pose.vertices.len()];
    let mut uses = Vec::new();
    let mut queue = VecDeque::new();
    queue.push_back(sink);
    while !queue.is_empty() {
        let i = queue.pop_front().unwrap();
        uses.push(i);
        if uses.len() == size {
            break;
        }
        for &to in &neighbors[i] {
            if visited[to] {
                continue;
            }
            visited[to] = true;
            queue.push_back(to);
        }
    }
    assert!(uses.len() == size);

    let dx = Binomial::new(16, 0.5).unwrap().sample(rng) as i64 - 8;
    let dy = Binomial::new(16, 0.5).unwrap().sample(rng) as i64 - 8;
    for &idx in &uses {
        let p = pose.vertices[idx];
        pose.vertices[idx] = Point(p.0 + dx, p.1 + dy);
    }
    *cache = NearestCache::new(pose, prob);
    let new_cost = cost(pose, prob, cache, temp);
    if new_cost <= old_cost {
        return true;
    }
    if rng.gen::<f64>() > ((old_cost - new_cost) / temp).exp() {
        for &idx in &uses {
            let p = pose.vertices[idx];
            pose.vertices[idx] = Point(p.0 - dx, p.1 - dy);
        }
        *cache = NearestCache::new(pose, prob);
    }
    false
}

fn move_all(
    pose: &mut Pose,
    prob: &Problem,
    rng: &mut SmallRng,
    temp: f64,
    cache: &mut NearestCache,
) -> bool {
    let old_cost = cost_unchecked(pose, prob, cache, temp);
    let dx = Binomial::new(16, 0.5).unwrap().sample(rng) as i64 - 8;
    let dy = Binomial::new(16, 0.5).unwrap().sample(rng) as i64 - 8;
    for p in &mut pose.vertices {
        *p = Point(p.0 + dx, p.1 + dy);
    }
    *cache = NearestCache::new(pose, prob);
    let new_cost = cost(pose, prob, cache, temp);
    if new_cost <= old_cost {
        return true;
    }
    if rng.gen::<f64>() > ((old_cost - new_cost) / temp).exp() {
        for p in &mut pose.vertices {
            *p = Point(p.0 - dx, p.1 - dy);
        }
        *cache = NearestCache::new(pose, prob);
    }
    false
}

fn rotate_all(
    pose: &mut Pose,
    prob: &Problem,
    rng: &mut SmallRng,
    temp: f64,
    cache: &mut NearestCache,
) -> bool {
    let old_cost = cost_unchecked(pose, prob, cache, temp);
    let rad = rng.gen::<f64>() - 0.5;
    let mut sx = 0;
    let mut sy = 0;
    for p in &pose.vertices {
        sx += p.0;
        sy += p.1;
    }
    let gx = sx as f64 / pose.vertices.len() as f64;
    let gy = sy as f64 / pose.vertices.len() as f64;
    let mut new_pose = pose.clone();
    for (idx, p) in pose.vertices.iter().enumerate() {
        let x = p.0 as f64 - gx;
        let y = p.1 as f64 - gy;
        let rx = rad.cos() * x - rad.sin() * y;
        let ry = rad.sin() * x + rad.cos() * y;
        new_pose.vertices[idx] = Point((rx + gx) as i64, (ry + gy) as i64);
    }
    let new_cache = NearestCache::new(&new_pose, prob);
    let new_cost = cost(&new_pose, prob, &new_cache, temp);
    if new_cost <= old_cost {
        pose.vertices = new_pose.vertices;
        *cache = new_cache;
        return true;
    }
    if rng.gen::<f64>() < ((old_cost - new_cost) / temp).exp() {
        pose.vertices = new_pose.vertices;
        *cache = new_cache;
    }
    false
}

fn gen_random_pose(prob: &Problem, rng: &mut SmallRng) -> Pose {
    let mut perm: Vec<_> = (0..prob.figure.vertices.len()).collect();
    let mut min_cost = 1e12;
    let mut current_pose = None;
    for _i in 0..10000 {
        let (left, _right) = perm.partial_shuffle(rng, prob.hole.len());

        let mut vertices = vec![*prob.hole.first().unwrap(); prob.figure.vertices.len()];

        for (i, &j) in left.iter().enumerate() {
            vertices[j] = prob.hole[i];
        }

        let pose = Pose { vertices };
        let cache = NearestCache::new(&pose, prob);
        let c = cost(&pose, prob, &cache, 1e4);
        if c < min_cost && pose.is_in_hole(&prob.figure.edges, &prob.hole) {
            current_pose = Some(pose);
            min_cost = c;
        }
    }
    match current_pose {
        Some(pose) => pose,
        None => Pose {
            vertices: vec![*prob.hole.first().unwrap(); prob.figure.vertices.len()],
        },
    }
}

fn solve(prob: &Problem, verbose: bool, loop_count: usize) -> Pose {
    let mut small_rng = SmallRng::from_entropy();

    let mut pose = gen_random_pose(prob, &mut small_rng);
    if dislike(&pose, prob) == 0 {
        return pose;
    }

    let start_temp: f64 = 1e6;
    let end_temp: f64 = 1e0;

    let mut improve_rorate_all = 0;
    let mut improve_move_all = 0;
    let mut improve_move_several = 0;
    let mut improve_flip_one = 0;
    let mut improve_move_one = 0;
    let mut improve_move_two = 0;
    let mut improve_rotate_two = 0;

    let mut cache = NearestCache::new(&pose, prob);
    for i in 0..loop_count {
        let ratio = i as f64 / loop_count as f64;
        let temp = (ratio * end_temp.ln() + (1.0 - ratio) * start_temp.ln()).exp();
        if i % 10000 == 0 && verbose {
            eprintln!(
                "{} {} {}",
                temp,
                cost(&pose, prob, &cache, temp),
                serde_json::to_string(&pose).unwrap()
            );
        }
        let rem = Uniform::from(0..64).sample(&mut small_rng);
        if rem <= 3 {
            if rotate_all(&mut pose, prob, &mut small_rng, temp, &mut cache) {
                improve_rorate_all += 1;
            }
        } else if rem <= 7 {
            if move_all(&mut pose, prob, &mut small_rng, temp, &mut cache) {
                improve_move_all += 1;
            }
        } else if rem <= 11 {
            if move_several(&mut pose, prob, &mut small_rng, temp, &mut cache) {
                improve_move_several += 1;
            }
        } else if rem <= 15 {
            if flip_one(&mut pose, prob, &mut small_rng, temp, &mut cache) {
                improve_flip_one += 1;
            }
        } else if rem <= 31 {
            if move_one(&mut pose, prob, &mut small_rng, temp, &mut cache) {
                improve_move_one += 1;
            }
        } else if rem <= 47 {
            if move_two(&mut pose, prob, &mut small_rng, temp, &mut cache) {
                improve_move_two += 1;
            }
        } else {
            if rotate_two(&mut pose, prob, &mut small_rng, temp, &mut cache) {
                improve_rotate_two += 1;
            }
        }
    }
    if verbose {
        eprintln!("[Stats] rotall: {}, movall: {}, movsev: {}, flipone: {}, movone: {}, movtwo: {}, rottwo: {}",
            improve_rorate_all, improve_move_all, improve_move_several, improve_flip_one, improve_move_one, improve_move_two, improve_rotate_two);
    }
    pose
}

fn dislike(pose: &Pose, prob: &Problem) -> u64 {
    if !pose.is_in_hole(&prob.figure.edges, &prob.hole) {
        return 1_000_000_000_000_000_000;
    }
    for &(u, v) in &prob.figure.edges {
        let orig_seg = Segment(prob.figure.vertices[u], prob.figure.vertices[v]);
        let pose_seg = Segment(pose.vertices[u], pose.vertices[v]);
        let ratio = pose_seg.length() as f64 / orig_seg.length() as f64;
        if (ratio - 1.0).abs() > prob.epsilon as f64 / 1e6 {
            return 1_000_000_000_000_000_000;
        }
    }
    let mut result = 0;
    for &p in &prob.hole {
        let mut min_d = None;
        for &q in &pose.vertices {
            min_d = Some(match min_d {
                Some(d) => min(d, p.sub(q).norm()),
                None => p.sub(q).norm(),
            });
        }
        result += min_d.unwrap() as u64;
    }
    result
}

fn command_solve(matches: &ArgMatches) -> std::io::Result<()> {
    let input_file = matches.value_of("problem").unwrap();
    let output_file = matches.value_of("answer").unwrap();
    let verbose = match matches.value_of("loglevel") {
        Some(level) => level.parse::<i32>().unwrap() > 1,
        None => false,
    };
    let default_loop_count = 500000;
    let loop_count = match matches.value_of("loop-count") {
        Some(s) => s.parse::<usize>().unwrap_or(default_loop_count),
        None => default_loop_count,
    };

    let prob = parse_problem(&input_file)?;

    let answer = solve(&prob, verbose, loop_count);

    println!("{}", dislike(&answer, &prob));

    let out_f = File::create(&output_file)?;
    let mut writer = BufWriter::new(out_f);
    write!(&mut writer, "{}", serde_json::to_string(&answer).unwrap())?;
    Ok(())
}

fn command_score(matches: &ArgMatches) -> std::io::Result<()> {
    let prob_file = matches.value_of("problem").unwrap();
    let ans_file = matches.value_of("answer").unwrap();

    let prob = parse_problem(&prob_file)?;
    let answer = parse_pose(&ans_file)?;

    println!("{}", dislike(&answer, &prob));
    Ok(())
}

fn main() -> std::io::Result<()> {
    let matches = App::new("ICFPC2021")
        .subcommand(
            SubCommand::with_name("solve")
                .arg(
                    Arg::with_name("problem")
                        .short("p")
                        .required(true)
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("answer")
                        .short("a")
                        .required(true)
                        .takes_value(true),
                )
                .arg(Arg::with_name("loglevel").short("l").takes_value(true))
                .arg(Arg::with_name("loop-count").short("n").takes_value(true)),
        )
        .subcommand(
            SubCommand::with_name("score")
                .arg(
                    Arg::with_name("problem")
                        .short("p")
                        .required(true)
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("answer")
                        .short("a")
                        .required(true)
                        .takes_value(true),
                ),
        )
        .get_matches();

    match matches.subcommand() {
        ("solve", Some(matches)) => command_solve(matches),
        ("score", Some(matches)) => command_score(matches),
        _ => {
            eprintln!("Unknown subcommand");
            Ok(())
        }
    }
}
