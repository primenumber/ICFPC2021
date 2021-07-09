use clap::{App, Arg, ArgMatches, SubCommand};
use rand::distributions::{Distribution, Uniform};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::Binomial;
use serde::{Deserialize, Serialize};
use std::cmp::min;
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader, BufWriter};
use std::mem;

#[derive(Clone, Copy, Deserialize, Serialize)]
struct Point(i64, i64);

impl Point {
    fn dot(&self, rhs: Point) -> i64 {
        self.0 * rhs.0 + self.1 * rhs.1
    }

    fn norm(&self) -> i64 {
        self.dot(*self)
    }

    fn cross(&self, rhs: Point) -> i64 {
        self.0 * rhs.1 - self.1 * rhs.0
    }

    fn sub(&self, rhs: Point) -> Point {
        Point(self.0 - rhs.0, self.1 - rhs.1)
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
        ccw(self.0, self.1, other.0) * ccw(self.0, self.1, other.1) == -1
            && ccw(other.0, other.1, self.0) * ccw(other.0, other.1, self.1) == -1
    }
}

#[derive(Serialize, Deserialize)]
struct Figure {
    vertices: Vec<Point>,
    edges: Vec<(usize, usize)>,
}

#[derive(Serialize, Deserialize)]
struct Problem {
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

fn cost(pose: &Pose, prob: &Problem, weight: f64) -> f64 {
    if !pose.is_in_hole(&prob.figure.edges, &prob.hole) {
        return 1e12;
    }
    let mut result = 0.0;
    for &(u, v) in &prob.figure.edges {
        let orig_seg = Segment(prob.figure.vertices[u], prob.figure.vertices[v]);
        let pose_seg = Segment(pose.vertices[u], pose.vertices[v]);
        let ratio = pose_seg.length() as f64 / orig_seg.length() as f64;
        result += (((ratio - 1.0).abs() * 1e6 - prob.epsilon as f64) * 1e5 / weight).max(0.0);
    }
    for &p in &prob.hole {
        let mut min_d = None;
        for &q in &pose.vertices {
            min_d = Some(match min_d {
                Some(d) => min(d, p.sub(q).norm()),
                None => p.sub(q).norm(),
            });
        }
        result += min_d.unwrap() as f64;
    }
    result
}

fn move_one(pose: &mut Pose, prob: &Problem, rng: &mut SmallRng, temp: f64) {
    let old_cost = cost(pose, prob, temp);
    let idx = Uniform::from(0..pose.vertices.len()).sample(rng);
    let p = pose.vertices[idx];
    let dx = Binomial::new(16, 0.5).unwrap().sample(rng) as i64 - 8;
    let dy = Binomial::new(16, 0.5).unwrap().sample(rng) as i64 - 8;
    let np = Point(p.0 + dx, p.1 + dy);
    pose.vertices[idx] = np;
    let new_cost = cost(pose, prob, temp);
    if new_cost <= old_cost {
        return;
    }
    if rng.gen::<f64>() > ((old_cost - new_cost) / temp).exp() {
        pose.vertices[idx] = p;
    }
}

fn move_all(pose: &mut Pose, prob: &Problem, rng: &mut SmallRng, temp: f64) {
    let old_cost = cost(pose, prob, temp);
    let dx = Binomial::new(16, 0.5).unwrap().sample(rng) as i64 - 8;
    let dy = Binomial::new(16, 0.5).unwrap().sample(rng) as i64 - 8;
    for p in &mut pose.vertices {
        *p = Point(p.0 + dx, p.1 + dy);
    }
    let new_cost = cost(pose, prob, temp);
    if new_cost <= old_cost {
        return;
    }
    if rng.gen::<f64>() > ((old_cost - new_cost) / temp).exp() {
        for p in &mut pose.vertices {
            *p = Point(p.0 - dx, p.1 - dy);
        }
    }
}

fn rotate_all(pose: &mut Pose, prob: &Problem, rng: &mut SmallRng, temp: f64) {
    let old_cost = cost(pose, prob, temp);
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
        new_pose.vertices[idx] = Point(rx as i64, ry as i64);
    }
    let new_cost = cost(&new_pose, prob, temp);
    if new_cost <= old_cost {
        pose.vertices = new_pose.vertices;
        return;
    }
    if rng.gen::<f64>() < ((old_cost - new_cost) / temp).exp() {
        pose.vertices = new_pose.vertices;
    }
}

fn solve(prob: &Problem, verbose: bool) -> Pose {
    let mut small_rng = SmallRng::from_entropy();

    let mut pose = Pose {
        vertices: vec![*prob.hole.first().unwrap(); prob.figure.vertices.len()],
    };

    let start_temp: f64 = 1e6;
    let end_temp: f64 = 1e3;
    let loop_count = 1000000;

    for i in 0..loop_count {
        let ratio = i as f64 / loop_count as f64;
        let temp = (ratio * end_temp.ln() + (1.0 - ratio) * start_temp.ln()).exp();
        if i % 10000 == 0 && verbose {
            eprintln!(
                "{} {} {}",
                temp,
                cost(&pose, prob, temp),
                serde_json::to_string(&pose).unwrap()
            );
        }
        if i % 10 == 0 {
            rotate_all(&mut pose, prob, &mut small_rng, temp);
        } else if i % 4 == 0 {
            move_all(&mut pose, prob, &mut small_rng, temp);
        } else {
            move_one(&mut pose, prob, &mut small_rng, temp);
        }
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

    let prob = parse_problem(&input_file)?;

    let answer = solve(&prob, verbose);

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
                .arg(Arg::with_name("loglevel").short("l").takes_value(true)),
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
