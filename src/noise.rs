use vector::vector::{VectorD, VectorS};
use std::f64;

fn lerp(v1: &VectorD, v2: &VectorD, w: f64) -> Result<VectorD, String> {
    let w = w * w * w * (6.0 * w * w - 15.0 * w + 10.0);
    v1.lerp(v2, w)
}

/// Normal Perlin Noise is Perlin
/// Barycentric is as shown in git link
pub enum NoiseType {
    Perlin, Barycentric
}

/// struct to handle generating perlin noise
pub struct PerlinNoise {
    noise_type: NoiseType,
    grad: Vec<Vec<VectorD>>,
    offsets: Vec<VectorD>,
    in_dim: usize, out_dim: usize, 
    bounds: VectorS
}

impl PerlinNoise {
    pub fn new(noise_type: NoiseType, _in_dim: usize, out_dim: usize, bounds: VectorS) -> Self {
        println!("currently in_dim is not utilized and is force set to 2");
        let mut noise = Self {
            grad: Vec::new(),
            offsets: Vec::new(),
            in_dim: 2, out_dim,
            noise_type, bounds
        };
        let _ = noise.regen();
        noise
    }
}

impl PerlinNoise {
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

    /// regenerate the random weights
    pub fn regen(&mut self) -> Result<(), String> {
        self.grad = Vec::new();
        self.offsets = Vec::new();
        let origin = VectorD::new(self.out_dim);
        let last_index = self.cumulative_bounds().pop().unwrap() + 1;

        for _ in 0..last_index {
            match self.noise_type {
                NoiseType::Perlin => {
                    let mut t = Vec::new();
                    for _ in 0..self.out_dim {
                        t.push(VectorD::rand(2));
                    }
                    let mut tt = Vec::new();

                    for j in 0..self.in_dim {
                        let mut v = Vec::new();
                        for i in 0..self.out_dim {
                            v.push(t[i][j]);
                        }
                        tt.push(VectorD::from(v));
                    }

                    self.grad.push(tt);
                    self.offsets.push(origin.clone());
                },
                NoiseType::Barycentric => {
                    let t = VectorD::rand(self.out_dim);

                    let mut max_l = f64::INFINITY;
                    let mut min_l = f64::NEG_INFINITY;
                    
                    // Find the range of values for l such that
                    // (l * t + simplexCenter) is still inside the 
                    // simplex. Recall the simplex is bounded by hyperplane
                    // x_i >= 0, so this is easy to calculate.
                    for i in 0..self.out_dim {
                        if t[i] != 0.0 {
                            let l = -1.0 / (self.out_dim as f64 * t[i]);
                            if l < 0.0 {
                                min_l = f64::max(min_l, l);
                            } else {
                                max_l = f64::min(max_l, l);
                            }
                        }
                    }
                    if max_l < min_l {
                        return Err(format!("invalid vectors"));
                    }
                    // Given min_l and max_l, we compute how to 
                    // scale and offset the gradient so that the range
                    // matches min_l and max_l
                    let center = (min_l + max_l) / 2.0;
                    let half_width = (max_l - min_l) / 2.0;
                    let offset = (&t * center).shift(1.0 / self.out_dim as f64);
                    let t = t * half_width;

                    let dir = VectorD::rand(self.out_dim);
                    let mut grad_x = Vec::new();

                    for i in 0..self.in_dim {
                        grad_x.push(&t * dir[i])
                    }
                    
                    self.grad.push(grad_x);
                    self.offsets.push(offset);
                }
            }
        }
        Ok(())
    }

    fn apply_grad(&self, ix: VectorD, x: VectorD) -> Result<VectorD, String> {
        let pos: VectorS = ix.convert(|x| x as usize);
        let pos = self.linearize(&pos);
        let delta = (&x - &ix)?;

        let g = &self.grad[pos];
        
        let total = g.iter().enumerate().fold(
            Ok(VectorD::new(self.out_dim)), 
            |sum , (i, gi)| {
                sum? + gi * delta[i]
            }
        );

        &self.offsets[pos] + &total?
    }

    /*
    fn apply_grad(&self, ix: usize, iy: usize, x: f64, y: f64) -> Result<VectorD, String> {
        let (dx, dy) = (x - ix as f64, y - iy as f64);
        let (dx, dy) = (dx as f64, dy as f64);
        let g = &self.grad[ix][iy];
        &self.offsets[ix][iy] + &(&g.0 * dx + &g.1 * dy)?
    }
    */

    /// evaluate the noise function at a point
    pub fn eval(&self, x: f64, y: f64) -> VectorD {
        // Determine grid cell coordinates
        let x0 = x.floor();
        let x1 = x0 + 1.0;
        let y0 = y.floor();
        let y1 = y0 + 1.0;
    
        // Determine interpolation weights
        // Could also use higher order polynomial/s-curve here
        let  sx = x - x0 as f64;
        let sy = y - y0 as f64;
        
        // Interpolate between grid point gradients
        let n0 = self.apply_grad(VectorD::from(vec![x0, y0]), VectorD::from(vec![x, y])).unwrap();
        let n1 = self.apply_grad(VectorD::from(vec![x1, y0]), VectorD::from(vec![x, y])).unwrap();
        let ix0 = lerp(&n0, &n1, sx).unwrap();

        let n0 = self.apply_grad(VectorD::from(vec![x0, y1]), VectorD::from(vec![x, y])).unwrap();
        let n1 = self.apply_grad(VectorD::from(vec![x1, y1]), VectorD::from(vec![x, y])).unwrap();
        let ix1 = lerp(&n0, &n1, sx).unwrap();
        
        let value = lerp(&ix0, &ix1, sy).unwrap();
    
        return value;
    }
}