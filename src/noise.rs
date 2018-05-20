use std::f64;

use num::traits::*;
use rand::{Rng, Rand, thread_rng};
use linear_algebra::vector::{Vector, VectorS};

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

/// struct to handle generating perlin noise
pub struct PerlinNoise<T>
    where T: Copy + Clone + Rand + Float {
    noise_type: NoiseType,
    grad: Vec<Vec<Vector<T>>>,
    offsets: Vec<Vector<T>>,
    in_dim: usize, out_dim: usize, 
    bounds: VectorS
}

impl<T> PerlinNoise<T>
    where T: Copy + Clone + Rand + Float {
    pub fn new(noise_type: NoiseType, in_dim: usize, out_dim: usize, bounds: VectorS) -> Result<Self, String> {
        let mut noise = Self {
            grad: Vec::new(),
            offsets: Vec::new(),
            in_dim, out_dim,
            noise_type, bounds
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
            let (a, b) = (p / x, p  % x);
            p = b;
            pos.insert(0, a);
        }

        VectorS::from(pos)
    }

    fn random_bary_vec(&self) -> Vector<T> {
        let mut rng = thread_rng();
        let n = self.out_dim;

        let zero = T::zero();
        let one = T::one();
        let two = one + one;
        
        loop {
            let mut v = Vec::new();
            let mut sum = zero;

            for _ in 0..n-1 {
                let x = rng.gen::<T>() * two - one;
                v.push(x);
                sum = sum + x;
            }

            v.push(zero - sum);
            let v = Vector::from(v);
            if v.magsq() <= one {
                return Vector::from(v).norm();
            }
        }
    }


    /// regenerate the random weights
    pub fn regen(&mut self) -> Result<(), String> {
        self.grad = Vec::new();
        self.offsets = Vec::new();
        let origin = Vector::new(self.out_dim);
        let last_index = self.cumulative_bounds().pop().unwrap() + 1;

        for _ in 0..last_index {
            match self.noise_type {
                NoiseType::Perlin => {
                    let mut t = Vec::new();
                    for _ in 0..self.out_dim {
                        t.push(Vector::rand(2));
                    }
                    let mut tt = Vec::new();

                    for j in 0..self.in_dim {
                        let mut v = Vec::new();
                        for i in 0..self.out_dim {
                            v.push(t[i][j]);
                        }
                        tt.push(Vector::from(v));
                    }

                    self.grad.push(tt);
                    self.offsets.push(origin.clone());
                },
                NoiseType::Barycentric => {
                    let t = self.random_bary_vec();

                    let mut max_l = T::infinity();
                    let mut min_l = T::neg_infinity();
                    let out_dim = T::from(self.out_dim).unwrap();
                    let one = T::one();
                    let two = one + one;
                    
                    // Find the range of values for l such that
                    // (l * t + simplexCenter) is still inside the 
                    // simplex. Recall the simplex is bounded by hyperplane
                    // x_i >= 0, so this is easy to calculate.
                    for i in 0..self.out_dim {
                        if !t[i].is_zero() {
                            let l = one.neg() / (out_dim * t[i]);
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
        }

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
            
            del[level] = T::zero();
            let n0 = self.eval_rec(v, v0, del, level)?;
            
            del[level] = T::one();
            let n1 = self.eval_rec(v, v0, del, level)?;

            lerp(&n0, &n1, v[level] - v0[level])
        } else {
            self.apply_grad(&(v0 + del)?, &v)
        }
    }

    pub fn eval(&self, v: &Vector<T>) -> Result<Vector<T>, String> {
        self.eval_rec(&v, &v.map(|&x| x.floor()), &mut vectorize![T::zero(); v.dim()], v.dim())
    }
}