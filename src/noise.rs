use rand::{Rng, ThreadRng, thread_rng};
use vector::Vector;
use std::f64;

fn lerp(v1: &Vector, v2: &Vector, w: f64) -> Result<Vector, String> {
    let w = w * w * w * (6.0 * w * w - 15.0 * w + 10.0);
    v1.lerp(v2, w)
}

pub enum NoiseType {
    Perlin, Barycentric
}

pub struct PerlinNoise {
    rng: ThreadRng, noise_type: NoiseType,
    grad: Vec<Vec<(Vector, Vector)>>,
    offsets: Vec<Vec<Vector>>,
    dim: usize, width: usize, height: usize
}

impl PerlinNoise {
    pub fn new(noise_type: NoiseType, dim: usize, width: usize, height: usize) -> Self {
        let mut noise = Self {
            rng: thread_rng(),
            grad: Vec::new(),
            offsets: Vec::new(),
            noise_type, dim, width, height
        };
        let _ = noise.regen();
        noise
    }
}

impl PerlinNoise {

}

impl PerlinNoise {
    pub fn regen(&mut self) -> Result<(), String> {
        self.grad = Vec::new();
        self.offsets = Vec::new();
        let origin = Vector::new(self.dim);

        for x in 0..self.width+1 {
            self.grad.push(Vec::new());
            self.offsets.push(Vec::new());
            for y in 0..self.height+1 {
                match self.noise_type {
                    NoiseType::Perlin => {
                        let mut t = Vec::new();
                        for _ in 0..self.dim {
                            t.push(Vector::rand(2));
                        }
                        let mut t0 = Vector::new(self.dim);
                        let mut t1 = Vector::new(self.dim);

                        for i in 0..self.dim {
                            t0[i] = t[i][0];
                            t1[i] = t[i][1];
                        }

                        self.grad[x][y] = (t0, t1);
                        self.offsets[x][y] = origin.clone();
                    },
                    NoiseType::Barycentric => {
                        let t = Vector::rand(self.dim);

                        let mut max_l = f64::INFINITY;
                        let mut min_l = f64::NEG_INFINITY;
                        
                        // Find the range of values for l such that
                        // (l * t + simplexCenter) is still inside the 
                        // simplex. Recall the simplex is bounded by hyperplane
                        // x_i >= 0, so this is easy to calculate.
                        for i in 0..self.dim {
                            if t[i] != 0.0 {
                                let l = -1.0 / (self.dim as f64 * t[i]);
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
                        let offset = (center * &t).shift(1.0 / self.dim as f64);
                        let t = t * half_width;

                        // Same as BARYCENTRIC_VARIANT
                        let angle = self.rng.gen::<f64>() * 2.0 * f64::consts::PI;
                        let s = angle.sin();
                        let c = angle.cos();
                        self.grad[x][y] = (s * &t, c * &t);
                        self.offsets[x][y] = offset;
                    }
                }
            }
        }
        Ok(())
    }

    fn apply_grad(&self, ix: usize, iy: usize, x: f64, y: f64) -> Result<Vector, String> {
        let (dx, dy) = (x - ix as f64, y - iy as f64);
        let (dx, dy) = (dx as f64, dy as f64);
        let g = &self.grad[ix][iy];
        &self.offsets[ix][iy] + &(dx * &g.0 + dy * &g.1)?
    }

    pub fn eval(&self, x: f64, y: f64) -> Vector {
        // Determine grid cell coordinates
        let x0 = x.floor() as usize;
        let x1 = x0 + 1;
        let y0 = y.floor() as usize;
        let y1 = y0 + 1;
    
        // Determine interpolation weights
        // Could also use higher order polynomial/s-curve here
        let  sx = x - x0 as f64;
        let sy = y - y0 as f64;
    
        // Interpolate between grid point gradients
        let n0 = self.apply_grad(x0, y0, x, y).unwrap();
        let n1 = self.apply_grad(x1, y0, x, y).unwrap();
        let ix0 = lerp(&n0, &n1, sx).unwrap();

        let n0 = self.apply_grad(x0, y1, x, y).unwrap();
        let n1 = self.apply_grad(x1, y1, x, y).unwrap();
        let ix1 = lerp(&n0, &n1, sx).unwrap();
        
        let value = lerp(&ix0, &ix1, sy).unwrap();
    
        return value;
    }
}