use clap::{App, Arg, ArgMatches, SubCommand};
use once_cell::sync::OnceCell;
use rand::distributions::{Distribution, Uniform};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_distr::{Binomial, Exp, Normal};
use serde::{Deserialize, Serialize};
use std::cmp::{max, min};
use std::collections::VecDeque;
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader, BufWriter};
use std::mem;

#[derive(PartialEq, Eq, Debug, Clone, Copy, Deserialize, Serialize)]
struct Point(i64, i64);

static USE_BONUS: OnceCell<Option<BonusType>> = OnceCell::new();
static WANT_BONUS: OnceCell<Vec<Bonus>> = OnceCell::new();

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

    fn reverse_x(&self) -> Point {
        Point(-self.0, self.1)
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

    fn is_in_hole(&self, hole: &[Point]) -> bool {
        for (idx, &p) in hole.iter().enumerate() {
            let q = hole[(idx + 1) % hole.len()];
            let t = Segment(p, q);
            if self.is_cross(t) {
                return false;
            }
            if ccw(self.0, self.1, q) == 0 {
                let r = hole[(idx + 2) % hole.len()];
                let qp = p.sub(q);
                let qr = r.sub(q);
                let qa = self.0.sub(q);
                let qb = self.1.sub(q);
                if qp.cross(qr) < 0 && q != self.0 && q != self.1 {
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
        true
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum BonusType {
    GLOBALIST,
    BREAK_A_LEG,
    WALLHACK,
    SUPERFLEX,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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
    epsilon: i64,
}

#[derive(Serialize, Deserialize, Clone)]
struct Pose {
    vertices: Vec<Point>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bonus: Option<BonusType>,
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
            if !s.is_in_hole(hole) {
                return false;
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
        let mut nearest_id = Vec::with_capacity(prob.hole.len());
        let mut nearest_distance = Vec::with_capacity(prob.hole.len());
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

    fn update_all(&mut self, pose: &Pose, prob: &Problem) {
        for (idx, &p) in prob.hole.iter().enumerate() {
            let mut min_d = 1_000_000_000_000_000_000;
            let mut min_id = 0;
            for (idx, &q) in pose.vertices.iter().enumerate() {
                let d = p.distance_sq(q);
                if d < min_d {
                    min_d = d;
                    min_id = idx;
                }
            }
            self.nearest_id[idx] = min_id;
            self.nearest_distance[idx] = min_d;
        }
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
    let mut max_xy = 0;
    for &p in &prob.hole {
        max_xy = max(max_xy, p.0);
        max_xy = max(max_xy, p.1);
    }
    let scale = (max_xy * max_xy) as f64;

    let mut result = 0.0;
    let mut max_diff = 0;
    for &(u, v) in &prob.figure.edges {
        let orig_len = prob.figure.vertices[u].distance_sq(prob.figure.vertices[v]);
        let pose_len = pose.vertices[u].distance_sq(pose.vertices[v]);
        let diff = max(
            0,
            1_000_000 * (pose_len - orig_len).abs() - prob.epsilon * orig_len,
        );
        if diff == 0 {
            continue;
        }
        max_diff = max(max_diff, diff);
        result += diff as f64 * scale / weight;
    }
    if pose.bonus == Some(BonusType::SUPERFLEX) {
        result -= max_diff as f64 * scale / weight;
    }
    result += cache.sum() as f64;
    let bonuses = WANT_BONUS.get().unwrap();
    for bonus in bonuses {
        if bonus.problem != 7 {
            continue;
        }
        if bonus.bonus == BonusType::SUPERFLEX {
            let mut min_d = 1_000_000_000_000_000_000;
            for &p in &pose.vertices {
                min_d = min(min_d, bonus.position.distance_sq(p));
            }
            result += (min_d * 1_000) as f64;
        }
    }
    result
}

fn cost(pose: &Pose, prob: &Problem, cache: &NearestCache, weight: f64) -> f64 {
    if !pose.is_in_hole(&prob.figure.edges, &prob.hole) {
        return 1e18;
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
    let dx = Normal::new(0.0, 4.0).unwrap().sample(rng) as i64;
    let dy = Normal::new(0.0, 4.0).unwrap().sample(rng) as i64;
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

fn move_one_mk2(
    pose: &mut Pose,
    prob: &Problem,
    rng: &mut SmallRng,
    temp: f64,
    cache: &mut NearestCache,
    neighbors: &[Vec<usize>],
) -> bool {
    let old_cost = cost_unchecked(pose, prob, cache, temp);
    let idx = Uniform::from(0..pose.vertices.len()).sample(rng);
    let mut shortest = 1_000_000;
    let mut short_id = idx;
    let fv = &prob.figure.vertices;
    for &ng in &neighbors[idx] {
        let orig_d = fv[idx].distance_sq(fv[ng]);
        let longest = (prob.epsilon + 1_000_000) * orig_d / 1_000_000;
        if longest < shortest {
            shortest = longest;
            short_id = ng;
        }
    }
    let q = fv[short_id];
    let mut candidates = Vec::new();
    let shortest_sqrt = (shortest as f64).sqrt().floor() as i64;
    for dx in -shortest_sqrt..=shortest_sqrt {
        let dymx = ((shortest - dx * dx) as f64).sqrt().floor() as i64;
        for dy in -dymx..=dymx {
            let p = q.add(Point(dx, dy));
            let mut ok = true;
            for &ng in &neighbors[idx] {
                let r = pose.vertices[ng];
                let d = p.distance_sq(r);
                let orig_d = fv[idx].distance_sq(fv[ng]);
                if 1_000_000 * (d - orig_d).abs() - prob.epsilon * orig_d > 0 {
                    ok = false;
                    break;
                }
            }
            if ok {
                candidates.push(p);
            }
        }
    }
    if candidates.is_empty() {
        return false;
    }
    let p = pose.vertices[idx];
    let np = *candidates.choose(rng).unwrap();
    if p == np {
        return false;
    }
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
    neighbors: &[Vec<usize>],
) -> bool {
    let old_cost = cost_unchecked(pose, prob, cache, temp);
    let mut flippables = Vec::new();
    for idx in 0..pose.vertices.len() {
        if neighbors[idx].len() == 2 {
            flippables.push(idx);
        }
    }
    if flippables.is_empty() {
        return false;
    }
    let idx = flippables[Uniform::from(0..flippables.len()).sample(rng)];

    let line = Segment(
        pose.vertices[neighbors[idx][0]],
        pose.vertices[neighbors[idx][1]],
    );
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
    let dx = Normal::new(0.0, 4.0).unwrap().sample(rng) as i64;
    let dy = Normal::new(0.0, 4.0).unwrap().sample(rng) as i64;
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
    neighbors: &[Vec<usize>],
) -> bool {
    let old_cost = cost_unchecked(pose, prob, cache, temp);
    let size = Uniform::from(2..=pose.vertices.len()).sample(rng);
    let sink = Uniform::from(0..pose.vertices.len()).sample(rng);
    let mut visited = vec![false; pose.vertices.len()];
    let mut uses = Vec::with_capacity(size);
    let mut queue = VecDeque::with_capacity(size);
    queue.push_back(sink);
    while !queue.is_empty() {
        let i = queue.pop_front().unwrap();
        uses.push(i);
        if uses.len() == size {
            break;
        }
        for &to in neighbors[i].choose_multiple(rng, neighbors[i].len()) {
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
    cache.update_all(pose, prob);
    let new_cost = cost(pose, prob, cache, temp);
    if new_cost <= old_cost {
        return true;
    }
    if rng.gen::<f64>() > ((old_cost - new_cost) / temp).exp() {
        for &idx in &uses {
            let p = pose.vertices[idx];
            pose.vertices[idx] = Point(p.0 - dx, p.1 - dy);
        }
        cache.update_all(pose, prob);
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
    cache.update_all(pose, prob);
    let new_cost = cost(pose, prob, cache, temp);
    if new_cost <= old_cost {
        return true;
    }
    if rng.gen::<f64>() > ((old_cost - new_cost) / temp).exp() {
        for p in &mut pose.vertices {
            *p = Point(p.0 - dx, p.1 - dy);
        }
        cache.update_all(pose, prob);
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
    let rad = Normal::new(0.0f64, 0.5).unwrap().sample(rng);
    let mut sx = 0;
    let mut sy = 0;
    for p in &pose.vertices {
        sx += p.0;
        sy += p.1;
    }
    let mut gx = sx as f64 / pose.vertices.len() as f64;
    let mut gy = sy as f64 / pose.vertices.len() as f64;
    let normal = Normal::new(0.0, 2.0).unwrap();
    gx += normal.sample(rng);
    gy += normal.sample(rng);
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

#[allow(dead_code)]
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

        let pose = Pose {
            bonus: None,
            vertices,
        };
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
            bonus: None,
            vertices: vec![*prob.hole.first().unwrap(); prob.figure.vertices.len()],
        },
    }
}

fn in_hole_points(prob: &Problem) -> Vec<Point> {
    let mut max_x = 0;
    let mut max_y = 0;
    for &p in &prob.hole {
        max_x = max(max_x, p.0);
        max_y = max(max_y, p.1);
    }
    let mut points = Vec::new();
    for x in 0..=max_x {
        for y in 0..=max_y {
            let p = Point(x, y);
            if p.is_in_hole(&prob.hole) {
                points.push(p);
            }
        }
    }
    points
}

#[allow(dead_code)]
fn gen_random_pose_mk2(prob: &Problem, rng: &mut SmallRng) -> Pose {
    let points = in_hole_points(prob);
    let mut vertices = vec![*prob.hole.first().unwrap(); prob.figure.vertices.len()];
    let mut min_cost = 1e12;
    let mut current_pose = None;
    for _i in 0..10000 {
        for v in &mut vertices {
            *v = *points.choose(rng).unwrap();
        }
        let pose = Pose {
            bonus: None,
            vertices: vertices.clone(),
        };
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
            bonus: None,
            vertices: vec![*prob.hole.first().unwrap(); prob.figure.vertices.len()],
        },
    }
}

fn gen_random_pose_mk3(prob: &Problem, rng: &mut SmallRng, verbose: bool, scale: f64) -> Pose {
    let points = in_hole_points(prob);
    let mut initial_pose = Pose {
        bonus: None,
        vertices: prob.figure.vertices.clone(),
    };
    let mut max_x = 0;
    let mut max_y = 0;
    for &p in &prob.hole {
        max_x = max(max_x, p.0);
        max_y = max(max_y, p.1);
    }
    let mut sx = 0;
    let mut sy = 0;
    for &p in &initial_pose.vertices {
        sx += p.0;
        sy += p.1;
    }
    let n = initial_pose.vertices.len();
    let g = Point(sx / n as i64, sy / n as i64);
    // transform and scale
    for p in &mut initial_pose.vertices {
        *p = p.sub(g).scale(scale);
    }
    let mut pose = initial_pose.clone();
    for _i in 0..100000 {
        let rad = rng.gen::<f64>() * 2.0 * (-1.0f64).acos();
        let offset = *points.choose(rng).unwrap();
        let flip = rng.gen::<bool>();
        for (idx, p) in initial_pose.vertices.iter_mut().enumerate() {
            pose.vertices[idx] = if flip {
                p.reverse_x().rotate(rad).add(offset)
            } else {
                p.rotate(rad).add(offset)
            };
        }
        if pose.is_in_hole(&prob.figure.edges, &prob.hole) {
            if verbose {
                eprintln!("gen mk3 scale: {}", scale);
            }
            return pose;
        }
    }
    gen_random_pose_mk3(prob, rng, verbose, scale * 0.9)
}

fn negibor_list(prob: &Problem) -> Vec<Vec<usize>> {
    let mut result = vec![Vec::new(); prob.figure.vertices.len()];
    for &(u, v) in &prob.figure.edges {
        result[u].push(v);
        result[v].push(u);
    }
    result
}

#[derive(Debug)]
struct ImproveCounter {
    rorate_all: usize,
    move_all: usize,
    move_several: usize,
    flip_one: usize,
    move_one: usize,
    move_one_mk2: usize,
    move_two: usize,
    rotate_two: usize,
}

fn improve_process(
    pose: &mut Pose,
    prob: &Problem,
    rng: &mut SmallRng,
    temp: f64,
    cache: &mut NearestCache,
    neighbors: &[Vec<usize>],
    counter: &mut ImproveCounter,
) {
    let rem = Uniform::from(0..64).sample(rng);
    if rem <= 3 {
        if rotate_all(pose, prob, rng, temp, cache) {
            counter.rorate_all += 1;
        }
    } else if rem <= 7 {
        if move_all(pose, prob, rng, temp, cache) {
            counter.move_all += 1;
        }
    } else if rem <= 11 {
        if move_several(pose, prob, rng, temp, cache, neighbors) {
            counter.move_several += 1;
        }
    } else if rem <= 15 {
        if flip_one(pose, prob, rng, temp, cache, neighbors) {
            counter.flip_one += 1;
        }
    } else if rem <= 31 {
        if move_one(pose, prob, rng, temp, cache) {
            counter.move_one += 1;
        }
    } else if rem <= 33 {
        if move_one_mk2(pose, prob, rng, temp, cache, neighbors) {
            counter.move_one_mk2 += 1;
        }
    } else if rem <= 47 {
        if move_two(pose, prob, rng, temp, cache) {
            counter.move_two += 1;
        }
    } else {
        if rotate_two(pose, prob, rng, temp, cache) {
            counter.rotate_two += 1;
        }
    }
}

fn solve(prob: &Problem, verbose: bool, loop_count: usize) -> Pose {
    let mut small_rng = SmallRng::from_entropy();

    let mut pose = gen_random_pose_mk3(prob, &mut small_rng, verbose, 1.0);
    if dislike(&pose, prob) == 0 {
        return pose;
    }

    let start_temp: f64 = 1e3;
    let end_temp: f64 = 3e-2;

    let mut counter = ImproveCounter {
        rorate_all: 0,
        move_all: 0,
        move_several: 0,
        flip_one: 0,
        move_one: 0,
        move_one_mk2: 0,
        move_two: 0,
        rotate_two: 0,
    };

    let neighbors = negibor_list(prob);
    let mut cache = NearestCache::new(&pose, prob);
    for i in 0..loop_count {
        let ratio = i as f64 / loop_count as f64;
        let temp = (ratio * end_temp.ln() + (1.0 - ratio) * start_temp.ln()).exp();
        improve_process(
            &mut pose,
            prob,
            &mut small_rng,
            temp,
            &mut cache,
            &neighbors,
            &mut counter,
        );
        if i % 50000 == 49999 && verbose {
            eprintln!(
                "{} {} {} {}",
                i,
                temp,
                cost(&pose, prob, &cache, temp),
                serde_json::to_string(&pose).unwrap()
            );
        }
    }

    if verbose {
        eprintln!("[Stats] {:?}", counter);
    }
    pose
}

fn improve(
    mut pose: Pose,
    prob: &Problem,
    matching: &[Option<usize>],
    matching_rev: &[Option<usize>],
    rng: &mut SmallRng,
    verbose: bool,
) -> Option<Pose> {
    if verbose {
        eprintln!("Start improve process: {:?}", matching);
    }
    let cache = NearestCache::new(&pose, prob);
    let mut old_cost = cost_unchecked(&pose, prob, &cache, 1e0);
    if old_cost == 0.0 {
        return Some(pose);
    }
    let mut rem_verts = Vec::new();
    for (idx, &m) in matching_rev.iter().enumerate() {
        if m == None {
            rem_verts.push(idx);
        }
    }
    let loop_count = 3000;
    for i in 0..loop_count {
        if i % 10000 == 9999 {
            eprintln!("{} {}", i, old_cost);
        }
        let mut new_pose = pose.clone();
        let size = Exp::new(1.0f64)
            .unwrap()
            .sample(rng)
            .min(rem_verts.len() as f64) as usize;
        for &idx in rem_verts.choose_multiple(rng, size) {
            if rng.gen::<f64>() < 0.5 {
                let dx = Normal::new(0.0, 1.0).unwrap().sample(rng) as i64;
                let dy = Normal::new(0.0, 1.0).unwrap().sample(rng) as i64;

                new_pose.vertices[idx].0 += dx;
                new_pose.vertices[idx].1 += dy;
            }
        }

        if !new_pose.is_in_hole(&prob.figure.edges, &prob.hole) {
            continue;
        }

        let new_cache = NearestCache::new(&new_pose, prob);

        let new_cost = cost_unchecked(&new_pose, prob, &new_cache, 1e0);
        if new_cost <= old_cost {
            pose = new_pose;
            old_cost = new_cost;
            if new_cost == 0.0 {
                return Some(pose);
            }
            continue;
        }

        let scale = 3e8 * (1.03 - (i as f64 / loop_count as f64).sqrt());
        if rng.gen::<f64>() < ((old_cost - new_cost) / scale).exp() {
            pose = new_pose;
            old_cost = new_cost;
            continue;
        }
    }
    None
}

fn dfs(
    i: usize,
    prob: &Problem,
    matrix: &[Vec<f64>],
    matrix_shortest: &[Vec<f64>],
    matching: &mut [Option<usize>],
    matching_rev: &mut [Option<usize>],
    rng: &mut SmallRng,
    verbose: bool,
) -> Option<Pose> {
    let m = prob.hole.len();
    let n = prob.figure.vertices.len();
    let fv = &prob.figure.vertices;
    let hv = &prob.hole;
    if i == m {
        let mut sx = 0;
        let mut sy = 0;
        for &p in &prob.hole {
            sx += p.0;
            sy += p.1;
        }
        let gx = sx / m as i64;
        let gy = sy / m as i64;
        let mut vertices = vec![Point(gx, gy); n];
        for (hidx, fidx) in matching.iter().enumerate() {
            vertices[fidx.unwrap()] = prob.hole[hidx];
        }
        let pose = Pose {
            bonus: *USE_BONUS.get().unwrap(),
            vertices,
        };
        improve(pose, prob, matching, matching_rev, rng, verbose)
    } else {
        for j in 0..n {
            if matching_rev[j] != None {
                continue;
            }
            matching[i] = Some(j);
            matching_rev[j] = Some(i);
            let mut valid = true;
            for &(u, v) in &prob.figure.edges {
                if matching_rev[u] == None || matching_rev[v] == None {
                    continue;
                }
                let uh = matching_rev[u].unwrap();
                let vh = matching_rev[v].unwrap();
                let s = Segment(hv[uh], hv[vh]);
                if !s.is_in_hole(&prob.hole) {
                    valid = false;
                    break;
                }
                let d = hv[uh].distance_sq(hv[vh]);
                let orig_d = fv[u].distance_sq(fv[v]);
                let ratio = d as f64 / orig_d as f64;
                if 1.0 - ratio > prob.epsilon as f64 * 1e-6 {
                    valid = false;
                    break;
                }
            }
            if !valid {
                matching[i] = None;
                matching_rev[j] = None;
                continue;
            }
            for k in 0..i {
                let kf = matching[k].unwrap();
                let d = hv[i].distance_sq(hv[k]);
                let longest = matrix[j][kf];
                if (d as f64).sqrt() > longest {
                    valid = false;
                    break;
                }
                let shortest = matrix_shortest[i][k];
                if longest < shortest {
                    valid = false;
                    break;
                }
            }
            if !valid {
                matching[i] = None;
                matching_rev[j] = None;
                continue;
            }
            match dfs(
                i + 1,
                prob,
                matrix,
                matrix_shortest,
                matching,
                matching_rev,
                rng,
                verbose,
            ) {
                Some(pose) => return Some(pose),
                None => (),
            };
            matching[i] = None;
            matching_rev[j] = None;
        }
        None
    }
}

fn solve_for_zero(prob: &Problem, verbose: bool, _loop_count: usize) -> Pose {
    let mut small_rng = SmallRng::from_entropy();
    let hv = &prob.hole;
    let fv = &prob.figure.vertices;
    let fe = &prob.figure.edges;
    let n = fv.len();
    let m = hv.len();
    if verbose {
        eprintln!("To solve for zero, N={}, M={}", n, m);
    }
    let mut matrix = vec![vec![f64::INFINITY; n]; n];
    for &(u, v) in fe {
        let max_dist =
            ((fv[u].distance_sq(fv[v]) as f64) * (1.0 + prob.epsilon as f64 * 1e-6)).sqrt();
        matrix[u][v] = max_dist;
        matrix[v][u] = max_dist;
    }
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                matrix[i][j] = matrix[i][j].min(matrix[i][k] + matrix[k][j]);
            }
        }
    }
    let mut matrix_shortest = vec![vec![f64::INFINITY; m]; m];
    for i in 0..m {
        matrix_shortest[i][i] = 0.0;
        for j in 0..m {
            let s = Segment(hv[i], hv[j]);
            if s.is_in_hole(hv) {
                matrix_shortest[i][j] = (s.length() as f64).sqrt();
            }
        }
    }
    for k in 0..m {
        for i in 0..m {
            for j in 0..m {
                matrix_shortest[i][j] =
                    matrix_shortest[i][j].min(matrix_shortest[i][k] + matrix_shortest[k][j]);
            }
        }
    }
    let mut matching = vec![None; hv.len()];
    let mut matching_rev = vec![None; n];
    match dfs(
        0,
        prob,
        &matrix,
        &matrix_shortest,
        &mut matching,
        &mut matching_rev,
        &mut small_rng,
        verbose,
    ) {
        Some(pose) => pose,
        None => Pose {
            bonus: None,
            vertices: vec![*hv.first().unwrap(); n],
        },
    }
}

fn dislike(pose: &Pose, prob: &Problem) -> u64 {
    if !pose.is_in_hole(&prob.figure.edges, &prob.hole) {
        return 1_000_000_000_000_000_000;
    }
    if pose.bonus == Some(BonusType::GLOBALIST) {
        let mut sum = 0.0;
        for &(u, v) in &prob.figure.edges {
            let orig_len =
                Segment(prob.figure.vertices[u], prob.figure.vertices[v]).length() as f64;
            let pose_len = Segment(pose.vertices[u], pose.vertices[v]).length() as f64;
            sum += (pose_len / orig_len - 1.0).abs().max(0.0);
        }
        if 1e6 * sum > (prob.figure.edges.len() * prob.epsilon as usize) as f64 {
            return 1_000_000_000_000_000_000;
        }
    } else {
        let mut count = 0;
        for &(u, v) in &prob.figure.edges {
            let orig_len = Segment(prob.figure.vertices[u], prob.figure.vertices[v]).length();
            let pose_len = Segment(pose.vertices[u], pose.vertices[v]).length();
            if 1_000_000 * (pose_len - orig_len).abs() > prob.epsilon * orig_len {
                if count > 0 || pose.bonus != Some(BonusType::SUPERFLEX) {
                    return 1_000_000_000_000_000_000;
                } else {
                    count += 1;
                }
            }
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

fn check_bonuses(pose: &Pose, prob: &Problem) -> Vec<Bonus> {
    let mut result = Vec::new();
    for bonus in &prob.bonuses {
        for &p in &pose.vertices {
            if p == bonus.position {
                result.push(bonus.clone());
            }
        }
    }
    result
}

fn command_solve(matches: &ArgMatches) -> std::io::Result<()> {
    let input_file = matches.value_of("problem").unwrap();
    let output_file = matches.value_of("answer").unwrap();
    let zero_desired = matches.is_present("zero");
    let use_superflex = matches.is_present("superflex");
    if use_superflex {
        USE_BONUS.set(Some(BonusType::SUPERFLEX)).unwrap();
    } else {
        USE_BONUS.set(None).unwrap();
    }
    let want_superflex = matches.is_present("want-superflex");
    let verbose = match matches.value_of("loglevel") {
        Some(level) => level.parse::<i32>().unwrap() > 1,
        None => false,
    };
    let default_loop_count = 5000000;
    let loop_count = match matches.value_of("loop-count") {
        Some(s) => s.parse::<usize>().unwrap_or(default_loop_count),
        None => default_loop_count,
    };

    let prob = parse_problem(&input_file)?;
    if want_superflex {
        let mut wants = Vec::new();
        for bonus in &prob.bonuses {
            if bonus.bonus == BonusType::SUPERFLEX {
                wants.push(bonus.clone());
            }
        }
        WANT_BONUS.set(wants).unwrap();
    } else {
        WANT_BONUS.set(Vec::new()).unwrap();
    }

    let mut answer = if zero_desired {
        solve_for_zero(&prob, verbose, loop_count)
    } else {
        solve(&prob, verbose, loop_count)
    };

    answer.bonus = USE_BONUS.get().unwrap().clone();

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

fn command_check_bonuses(matches: &ArgMatches) -> std::io::Result<()> {
    let prob_file = matches.value_of("problem").unwrap();
    let ans_file = matches.value_of("answer").unwrap();

    let prob = parse_problem(&prob_file)?;
    let answer = parse_pose(&ans_file)?;

    println!("{:?}", check_bonuses(&answer, &prob));
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
                .arg(Arg::with_name("zero").short("z"))
                .arg(Arg::with_name("superflex").short("s"))
                .arg(Arg::with_name("want-superflex").short("w"))
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
        .subcommand(
            SubCommand::with_name("check-bonuses")
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
        ("check-bonuses", Some(matches)) => command_check_bonuses(matches),
        _ => {
            eprintln!("Unknown subcommand");
            Ok(())
        }
    }
}
