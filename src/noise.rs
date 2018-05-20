use std::f64;

use num::traits::*;
use rand::{Rng, Rand, thread_rng};
use linear_algebra::vector::{Vector, VectorS};

#[derive(Clone, Copy)]
pub enum Mode {
    Debug, Release
}

macro_rules! ifdebug {
    ($on: expr => $thing: expr) => {
        let mode: Mode = $on;
        if let Mode::Debug = mode {
            $thing;
        }
    };
}

fn lerp<T>(v1: &Vector<T>, v2: &Vector<T>, w: T) -> Result<Vector<T>, String>
where T: Float {
    let a = T::from(6.0).unwrap();
    let b = T::from(15.0).unwrap();
    let c = T::from(10.0).unwrap();
    let w = w * w * w * (a * w * w - b * w + c);
    v1.lerp(v2, w)
}

/// Normal Perlin Noise is Perlin
/// Barycentric is as shown in git link
pub enum NoiseType {
    Perlin, Barycentric
}

pub enum Range {
    Pos,
    PosNeg
}

/// struct to handle generating perlin noise
pub struct PerlinNoise<T>
    where T: Copy + Clone + Rand + Float {
    noise_type: NoiseType,
    grad: Vec<Vec<Vector<T>>>,
    offsets: Vec<Vector<T>>,
    in_dim: usize, out_dim: usize, 
    bounds: VectorS, mode: Mode,
    range: Range,
    zero: T, one: T, two: T
}

impl<T> PerlinNoise<T>
    where T: Copy + Clone + Rand + Float {
    pub fn new(noise_type: NoiseType, in_dim: usize, out_dim: usize, bounds: VectorS, mode: Mode, range: Range) -> Result<Self, String> {
        let mut noise = Self {
            grad: Vec::new(),
            offsets: Vec::new(),
            in_dim, out_dim,
            noise_type, bounds,
            mode, range, zero: T::zero(),
            one: T::one(), two: (T::one() + T::one())
        };
        noise.regen()?;
        Ok(noise)
    }
}

impl<T> PerlinNoise<T>
    where T: Copy + Clone + Rand + Float {
    pub fn bounds(&self) -> &VectorS {
        &self.bounds
    }

    fn cumulative_bounds(&self) -> Vec<usize> {
        let mut v = vec![1];

        for i in 0..self.bounds.dim() {
            let x = *v.last().unwrap();
            let b = self.bounds[i] + 1;
            v.push(x * b);
        }

        v
    }

    fn linearize(&self, x: &VectorS) -> usize {
        VectorS::from(self.cumulative_bounds()).dot(x)
    }
    
    #[allow(dead_code)]
    fn vectorize(&self, mut p: usize) -> VectorS {
        let mut v = self.cumulative_bounds();
        let mut pos = vec![];

        while let Some(x) = v.pop() {
            let (a, b) = (p / x, p % x);
            p = b;
            pos.insert(0, a);
        }

        VectorS::from(pos)
    }

    fn random_bary_vec(&self) -> Vector<T> {
        let mut rng = thread_rng();
        let n = self.out_dim;

        let counter = 0usize;
        
        loop {
            ifdebug! (self.mode => {
                if counter % 1000 == 0 {
                    println!("rand finder = {}", counter)
                }
            });
            let mut v = Vec::new();
            let mut sum = self.zero;

            for _ in 0..n-1 {
                let x = rng.gen::<T>() * self.two - self.one;
                v.push(x);
                sum = sum + x;
            }

            v.push(self.zero - sum);
            let v = Vector::from(v);
            if v.magsq() <= self.one {
                return Vector::from(v).norm();
            }
        }
    }


    /// regenerate the random weights
    pub fn regen(&mut self) -> Result<(), String> {
        let last_index = self.cumulative_bounds().pop().unwrap() + 1;
        self.grad = Vec::with_capacity(last_index);
        self.offsets = Vec::with_capacity(last_index);
        let origin = Vector::new(self.out_dim);
        let perc = (last_index as f64 * 0.01) as usize + 1;

        ifdebug!(self.mode => println!("start regen"));

        
        match self.noise_type {
            NoiseType::Perlin => {
                ifdebug!(self.mode => println!("using perlin noise"));
                for i in 0..last_index {
                    let mut t = Vec::new();

                    for _ in 0..self.out_dim {
                        t.push(Vector::rand(self.in_dim));
                    }
                    let mut tt = Vec::new();

                    ifdebug!(self.mode => if i % perc == 0 { println!("generating gradient {} of {}", i, last_index) } );
                    for j in 0..self.in_dim {
                        let mut v = Vec::new();
                        for i in 0..self.out_dim {
                            v.push(t[i][j]);
                        }
                        tt.push(Vector::from(v));
                    }

                    self.grad.push(tt);
                    self.offsets.push(origin.clone());
                }
            },
            NoiseType::Barycentric => {
                ifdebug!(self.mode => println!("using barycentric noise"));
                let out_dim = T::from(self.out_dim).unwrap();
                let one = T::one();
                let two = one + one;

                for i in 0..last_index {
                    let t = self.random_bary_vec();

                    let mut max_l = T::infinity();
                    let mut min_l = T::neg_infinity();
                    
                    // Find the range of values for l such that
                    // (l * t + simplexCenter) is still inside the 
                    // simplex. Recall the simplex is bounded by hyperplane
                    // x_i >= 0, so this is easy to calculate.
                    ifdebug!(self.mode => if i % perc == 0 { println!("generating gradient {} of {}", i, last_index) } );
                    for i in 0..self.out_dim {
                        if !t[i].is_zero() {
                            let l = self.one.neg() / (out_dim * t[i]);
                            if l.is_sign_negative() {
                                min_l = min_l.max(l);
                            } else {
                                max_l = max_l.min(l);
                            }
                        }
                    }
                    if max_l < min_l {
                        return Err(format!("invalid vectors"));
                    }
                    // Given min_l and max_l, we compute how to 
                    // scale and offset the gradient so that the range
                    // matches min_l and max_l
                    let center = (min_l + max_l) / two;
                    let half_width = (max_l - min_l) / two;
                    let offset = (&t * &center).shift(T::from(1.0 / self.out_dim as f64).unwrap());
                    let t = t * half_width;

                    let dir: Vector<T> = Vector::rand(self.in_dim);
                    let mut grad_x = Vec::new();

                    for i in 0..self.in_dim {
                        grad_x.push(&t * &dir[i])
                    }
                    
                    self.grad.push(grad_x);
                    self.offsets.push(offset);
                }
            }
        };

        Ok(())
    }

    fn apply_grad(&self, ix: &Vector<T>, x: &Vector<T>) -> Result<Vector<T>, String> {
        let pos: VectorS = ix.map(|x| x.to_usize().unwrap());
        let pos = self.linearize(&pos);
        let delta = (x - ix)?;

        let g = &self.grad[pos];
        
        let total = g.iter().enumerate().fold(
            Ok(Vector::new(self.out_dim)), 
            |sum , (i, gi)| {
                sum? + gi * &delta[i]
            }
        );
        
        &self.offsets[pos] + &total?
    }

    pub fn eval_rec(&self, v: &Vector<T>, v0: &Vector<T>, del: &mut Vector<T>, level: usize) -> Result<Vector<T>, String> {
        if level > 0 {
            let level = level - 1;
            
            del[level] = self.zero;
            let n0 = self.eval_rec(v, v0, del, level)?;
            
            del[level] = self.one;
            let n1 = self.eval_rec(v, v0, del, level)?;

            lerp(&n0, &n1, v[level] - v0[level])
        } else {
            self.apply_grad(&(v0 + del)?, &v)
        }
    }

    pub fn eval(&self, v: &Vector<T>) -> Result<Vector<T>, String> {
        let eval = self.eval_rec(&v, &v.map(|&x| x.floor()), &mut vectorize![self.zero; v.dim()], v.dim())?;
        Ok(match self.range {
            Range::Pos => eval.shift(self.one),
            Range::PosNeg => eval * self.two
        })
    }
}