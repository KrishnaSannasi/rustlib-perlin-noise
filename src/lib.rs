/**
 * Author: Krishna Sannasi
 * 
 * Credits:
 * this is a rust implementation of https://github.com/BorisTheBrave/barycentric-perlin-noise/blob/master/perlin.ts
 */

#[macro_use]
extern crate linear_algebra;
extern crate rand;
extern crate num;

pub mod noise;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
