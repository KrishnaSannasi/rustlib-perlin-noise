use std::ops::{Add, Sub, Mul, Div, Index, IndexMut};
use rand::{Rng, thread_rng};

#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    value: Vec<f64>

}

// related functions
impl Vector {
    /// creates a vector of 0.0s with the dimension given
    pub fn new(dim: usize) -> Self {
        Self::from(vec![0.0; dim])
    }

    /// creates a random unit vector with the dimension given
    pub fn rand(dim: usize) -> Self {
        let mut vec = Vec::new();
        let mut rng = thread_rng();

        for _ in 0..dim {
            vec.push(rng.gen());
        }

        Self::from(vec).norm()
    }
}

// immutable methods
impl Vector {
    /// get the dimension of the vector
    pub fn dim(&self) -> usize {
        self.value.len()
    }

    /// the square of the magnitude
    pub fn magsq(&self) -> f64 {
        self.dot(self)
    }

    /// the magnitude
    pub fn mag(&self) -> f64 {
        self.magsq().sqrt()
    }

    /// returns a unit vector with the same 
    /// direction and dimension as the parent vector
    pub fn norm(&self) -> Self {
        self / self.mag()
    }

    /// sums up the elements of the vector
    pub fn sum(&self) -> f64 {
        let mut sum = 0.0;
        for i in self.value.iter() {
            sum += i;
        }
        sum
    }
    
    /// takes the dot product of the two vectors
    pub fn dot(&self, other: &Self) -> f64 {
        let mut dot = 0.0;
        
        for (i, j) in self.value.iter().zip(other.value.iter()) {
            dot += i * j;
        }

        dot
    }

    /// gives the angle between two vectors
    pub fn angle(&self, other: &Self) -> f64 {
        let y = self.dot(other);
        let x = (self.magsq() * other.magsq()).sqrt();

        (y / x).acos()
    }

    /// adds the shift value to all the elements in a vector
    pub fn shift(&self, value: f64) -> Self {
        let mut vec = Vec::new();

        for i in self.value.iter() {
            vec.push(i + value);
        }

        Self::from(vec)
    }

    /// linearly interpolates between two vectors
    pub fn lerp(&self, other: &Vector, w: f64) -> Result<Self, String> {
        self * (1.0 - w) + other * w
    }
}

// traits
impl<'a> From<&'a [f64]> for Vector {
    fn from(value: &'a [f64]) -> Self {
        Self::from(Vec::from(value))
    }
}

impl From<Vec<f64>> for Vector {
    fn from(value: Vec<f64>) -> Self {
        Self {
            value
        }
    }
}

impl Index<usize> for Vector {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.value[index]
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut f64 {
        &mut self.value[index]
    }

}

impl Add<Vector> for Vector {
    type Output = Result<Vector, String>;

    fn add(self, rhs: Vector) -> Self::Output {
        &self + &rhs
    }
}

impl Sub<Vector> for Vector {
    type Output = Result<Vector, String>;

    fn sub(self, rhs: Vector) -> Self::Output {
        &self - &rhs
    }
}

impl Mul<Vector> for Vector {
    type Output = Result<Vector, String>;

    fn mul(self, rhs: Vector) -> Self::Output {
        &self * &rhs
    }
}

impl Div<Vector> for Vector {
    type Output = Result<Vector, String>;

    fn div(self, rhs: Vector) -> Self::Output {
        &self / &rhs
    }
}

impl Mul<f64> for Vector {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        &self * rhs
    }
}

impl Div<f64> for Vector {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        &self / rhs
    }
}

impl Mul<Vector> for f64 {
    type Output = Vector;

    fn mul(self, rhs: Vector) -> Self::Output {
        &rhs * self
    }
}

impl<'a, 'b> Add<&'b Vector> for &'a Vector {
    type Output = Result<Vector, String>;

    fn add(self, rhs: &'b Vector) -> Self::Output {
        if self.dim() != rhs.dim() {
            return Err(format!("incompatible sizes"))
        }

        let mut vec = Vec::new();

        for (i, j) in self.value.iter().zip(rhs.value.iter()) {
            vec.push(i + j)
        }

        Ok(Vector::from(vec))
    }
}

impl<'a, 'b> Sub<&'b Vector> for &'a Vector {
    type Output = Result<Vector, String>;

    fn sub(self, rhs: &'b Vector) -> Self::Output {
        if self.dim() != rhs.dim() {
            return Err(format!("incompatible sizes"))
        }

        let mut vec = Vec::new();

        for (i, j) in self.value.iter().zip(rhs.value.iter()) {
            vec.push(i - j)
        }

        Ok(Vector::from(vec))
    }
}

impl<'a, 'b> Mul<&'b Vector> for &'a Vector {
    type Output = Result<Vector, String>;

    fn mul(self, rhs: &'b Vector) -> Self::Output {
        if self.dim() != rhs.dim() {
            return Err(format!("incompatible sizes"))
        }

        let mut vec = Vec::new();

        for (i, j) in self.value.iter().zip(rhs.value.iter()) {
            vec.push(i * j)
        }

        Ok(Vector::from(vec))
    }
}

impl<'a, 'b> Div<&'b Vector> for &'a Vector {
    type Output = Result<Vector, String>;

    fn div(self, rhs: &'b Vector) -> Self::Output {
        if self.dim() != rhs.dim() {
            return Err(format!("incompatible sizes"))
        }

        let mut vec = Vec::new();

        for (i, j) in self.value.iter().zip(rhs.value.iter()) {
            vec.push(i / j)
        }

        Ok(Vector::from(vec))
    }
}

impl<'a> Mul<f64> for &'a Vector {
    type Output = Vector;

    fn mul(self, rhs: f64) -> Self::Output {
        let mut vec = Vec::new();

        for i in self.value.iter() {
            vec.push(i * rhs)
        }

        Vector::from(vec)
    }
}

impl<'a> Div<f64> for &'a Vector {
    type Output = Vector;

    fn div(self, rhs: f64) -> Self::Output {
        let mut vec = Vec::new();

        for i in self.value.iter() {
            vec.push(i / rhs)
        }

        Vector::from(vec)
    }
}

impl<'a> Mul<&'a Vector> for f64 {
    type Output = Vector;

    fn mul(self, rhs: &'a Vector) -> Self::Output {
        rhs * self
    }
}
