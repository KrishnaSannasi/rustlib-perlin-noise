use std::f64;

use num::traits::*;
use rand::{Rng, Rand, thread_rng};
use linear_algebra::vector_sized::{Vector, Vectorizable};
use typenum::{Unsigned, NonZero};
use std::convert::TryFrom;

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

use std::fmt::Debug;

fn lerp<T, S: Unsigned>(v1: &Vector<T, S>, v2: &Vector<T, S>, w: T) -> Vector<T, S>
where T: Float + Debug + Vectorizable {
    let a = T::from(6.0).unwrap();
    let b = T::from(15.0).unwrap();
    let c = T::from(10.0).unwrap();
    let w = w * w * w * (a * w * w - b * w + c);
    v1.lerp(v2, w)
}

fn isneg<T>(x: T) -> usize
where T: Float {
    if x != T::zero() && x.is_sign_negative() {
        1
    } else {
        0
    }
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
pub struct PerlinNoise<T: Copy + Rand + Vectorizable + Float + Debug,
                       I: Unsigned + NonZero,
                       O: Unsigned + NonZero> {
    noise_type: NoiseType,
    grad: Vec<Vec<Vector<T, O>>>,
    offsets: Vec<Vector<T, O>>,
    bounds: Vector<usize, I>, mode: Mode,
    range: Range,
    zero: T, one: T, two: T
}

impl<T: Copy + Rand + Vectorizable + Float + Debug,
     I: Unsigned + NonZero,
     O: Unsigned + NonZero> PerlinNoise<T, I, O> {
    pub fn new(noise_type: NoiseType, bounds: Vector<usize, I>, mode: Mode, range: Range) -> Result<Self, String> {
        let mut noise = Self {
            grad: Vec::new(),
            offsets: Vec::new(),
            noise_type, bounds,
            mode, range, zero: T::zero(),
            one: T::one(), two: (T::one() + T::one())
        };
        noise.regen()?;
        Ok(noise)
    }
}

impl<T: Copy + Rand + Vectorizable + Float + Debug,
     I: Unsigned + NonZero,
     O: Unsigned + NonZero> PerlinNoise<T, I, O> {
    pub fn bounds(&self) -> &Vector<usize, I> {
        &self.bounds
    }

    pub fn in_bounds(&self, v: &Vector<T, I>) -> bool {
        if v.map(|&x| isneg(x)).sum() > 0 {
            false
        } else {
            let a = &self.bounds.map(|&x| T::from(x).unwrap()) - v;

            a.map(|&x| isneg(x)).sum() == 0
        }
    }

    fn cumulative_bounds(&self) -> Vec<usize> {
        let mut v = vec![1];

        for i in 0..self.bounds.dim() {
            let x = v[v.len() - 1];
            let b = self.bounds[i] + 1;
            v.push(x * b);
        }

        v
    }

    fn linearize(&self, x: &Vector<usize, I>) -> usize {
        let mut b = self.cumulative_bounds();
        b.pop();
        Vector::try_from(b).unwrap().dot(x)
    }
    
    fn random_bary_vec(&self) -> Vector<T, O> {
        let mut rng = thread_rng();
        let n = O::to_usize();

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
            let v = Vector::try_from(v).unwrap();
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
        let perc = (last_index as f64 * 0.01) as usize + 1;

        ifdebug!(self.mode => println!("start regen"));

        
        match self.noise_type {
            NoiseType::Perlin => {
                ifdebug!(self.mode => println!("using perlin noise"));
                for i in 0..last_index {
                    let mut t: Vec<Vector<T, I>> = Vec::new();

                    for _ in 0..O::to_usize() {
                        t.push(Vector::rand());
                    }
                    let mut tt = Vec::new();

                    ifdebug!(self.mode => if i % perc == 0 { println!("generating gradient {} of {}", i, last_index) } );
                    for j in 0..I::to_usize() {
                        let mut v = Vec::new();
                        for i in 0..O::to_usize() {
                            v.push(t[i][j]);
                        }
                        tt.push(Vector::try_from(v).unwrap());
                    }

                    self.grad.push(tt);
                    self.offsets.push(Vector::new());
                }
            },
            NoiseType::Barycentric => {
                ifdebug!(self.mode => println!("using barycentric noise"));
                let out_dim = T::from(O::to_usize()).unwrap();
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
                    for i in 0..O::to_usize() {
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
                    let offset = (&t * center).shift(T::from(1.0 / O::to_usize() as f64).unwrap());
                    let t = t * half_width;

                    let dir: Vector<T, I> = Vector::rand();
                    let mut grad_x = Vec::new();

                    for i in 0..I::to_usize() {
                        grad_x.push(&t * dir[i])
                    }
                    
                    self.grad.push(grad_x);
                    self.offsets.push(offset);
                }
            }
        };

        Ok(())
    }

    fn apply_grad(&self, ix: &Vector<T, I>, x: &Vector<T, I>) -> Vector<T, O> {
        let pos = ix.map(|x| match x.to_usize() {
            Some(val) => val,
            None => {
                panic!("out of bounds exception: input is negative")
            }
        });
        
        let pos = self.linearize(&pos);
        
        let delta = x - ix;

        let g = &self.grad[pos];
        
        let total = g.iter().enumerate().fold(
            Vector::new(), 
            |sum , (i, gi)| {
                sum + gi * delta[i]
            }
        );
        
        &self.offsets[pos] + &total
    }

    pub fn eval_rec(&self, v: &Vector<T, I>, v0: &Vector<T, I>, del: &mut Vector<T, I>, level: usize) -> Vector<T, O> {
        if level > 0 {
            let level = level - 1;
            
            del[level] = self.zero;
            let n0 = self.eval_rec(v, v0, del, level);
            
            del[level] = self.one;
            let n1 = self.eval_rec(v, v0, del, level);

            lerp(&n0, &n1, v[level] - v0[level])
        } else {
            let del: &Vector<_, _> = del;
            self.apply_grad(&(v0 + del), &v)
        }
    }

    pub fn eval(&self, v: &Vector<T, I>) -> Result<Vector<T, O>, String> {
        if self.in_bounds(v) {
            let mut del = Vector::new();
            let eval = self.eval_rec(&v, &v.map(|&x| x.floor()), &mut del, v.dim());
            Ok(match self.range {
                Range::Pos => eval.shift(self.two.recip()),
                Range::PosNeg => eval * self.two
            })   
        } else {
            Err(format!("{:?} is out of bounds! bounds are {:?}", v, self.bounds))
        }
    }
}