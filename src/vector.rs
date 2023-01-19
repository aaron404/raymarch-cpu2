use std::{ops::{Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign}, iter::Sum};


// #[derive(Default)]
#[derive(Copy, Clone, Debug)]
pub struct Vector<const N: usize, T> where T: Default + Copy + Clone {
    pub vals: [T; N],
}

pub trait Value: 
    Add + AddAssign + Sub + SubAssign + Mul + MulAssign + Div + DivAssign +
    Default + Sized + Copy + Clone {}
    
impl Value for f32 {}
impl Value for i32 {}
impl Value for u32 {}
impl Value for f64 {}

impl<const N: usize, T> Vector<N, T> where T: Value {
    const fn new(vals: [T; N]) -> Self {
        assert!(N > 0);
        Vector { vals }
    }
}

impl<const N: usize, T> Add for Vector<N, T> where T: Add<Output=T> + Value  {
    type Output = Vector<N, T>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut vals: [T; N] = [T::default(); N];
        for (i, sum) in self.vals.iter().zip(rhs.vals.iter()).enumerate() {
            vals[i] = *sum.0 + *sum.1;
        }
        Vector { vals }
    }
}

impl<const N: usize, T> AddAssign for Vector<N, T> where T: Value {
    fn add_assign(&mut self, rhs: Self) {
        for (i, val) in rhs.vals.iter().enumerate() {
            self.vals[i] += *val;
        }
    }
}

impl<const N: usize, T> Add<T> for Vector<N, T> where T: Value + Add<Output = T> {
    type Output = Vector<N, T>;

    fn add(self, rhs: T) -> Self::Output {
        let mut vals = [T::default(); N];
        vals.iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = self.vals[i] + rhs);
        Vector { vals }
    }
}

impl<const N: usize, T> AddAssign<T> for Vector<N, T> where T: Value + Add<Output = T> {
    fn add_assign(&mut self, rhs: T) {
        self.vals.iter_mut()
            .for_each(|v| *v += rhs);
    }
}

impl<const N: usize, T> Sub for Vector<N, T> where T: Sub<Output=T> + Value  {
    type Output = Vector<N, T>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut vals: [T; N] = [T::default(); N];
        for (i, sum) in self.vals.iter().zip(rhs.vals.iter()).enumerate() {
            vals[i] = *sum.0 - *sum.1;
        }
        Vector { vals }
    }
}

impl<const N: usize, T> Sub<T> for Vector<N, T> where T: Value + Sub<Output = T> {
    type Output = Vector<N, T>;

    fn sub(self, rhs: T) -> Self::Output {
        let mut vals = [T::default(); N];
        vals.iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = self.vals[i] - rhs);
        Vector { vals }
    }
}

impl<const N: usize, T> SubAssign for Vector<N, T> where T: Value {
    fn sub_assign(&mut self, rhs: Self) {
        for (i, val) in rhs.vals.iter().enumerate() {
            self.vals[i] -= *val;
        }
    }
}

impl<const N: usize, T> Mul for Vector<N, T> where T: Mul<Output=T> + Value  {
    type Output = Vector<N, T>;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut vals: [T; N] = [T::default(); N];
        for (i, sum) in self.vals.iter().zip(rhs.vals.iter()).enumerate() {
            vals[i] = *sum.0 * *sum.1;
        }
        Vector { vals }
    }
}

impl<const N: usize, T> Mul<T> for Vector<N, T> where T: Mul<Output=T> + Value  {
    type Output = Vector<N, T>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut vals = [T::default(); N];
        vals.iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = self.vals[i] * rhs);
        Vector { vals }
    }
}

impl<const N: usize> Mul<Vector<N, f32>> for f32 {
    type Output = Vector<N, f32>;

    fn mul(self, rhs: Vector<N, f32>) -> Self::Output {
        let mut vals = [f32::default(); N];
        vals.iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = self * rhs.vals[i]);
        Vector { vals }
    }
}

impl<const N: usize> Add<Vector<N, f32>> for f32 {
    type Output = Vector<N, f32>;

    fn add(self, rhs: Vector<N, f32>) -> Self::Output {
        let mut vals = [f32::default(); N];
        vals.iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = self + rhs.vals[i]);
        Vector { vals }
    }
}

impl<const N: usize> Sub<Vector<N, f32>> for f32 {
    type Output = Vector<N, f32>;

    fn sub(self, rhs: Vector<N, f32>) -> Self::Output {
        let mut vals = [f32::default(); N];
        vals.iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = self - rhs.vals[i]);
        Vector { vals }
    }
}

impl<const N: usize> Div<Vector<N, f32>> for f32 {
    type Output = Vector<N, f32>;

    fn div(self, rhs: Vector<N, f32>) -> Self::Output {
        let mut vals = [f32::default(); N];
        vals.iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = self / rhs.vals[i]);
        Vector { vals }
    }
}

impl<const N: usize, T> Div for Vector<N, T> where T: Div<Output=T> + Value  {
    type Output = Vector<N, T>;

    fn div(self, rhs: Self) -> Self::Output {
        let mut vals: [T; N] = [T::default(); N];
        for (i, sum) in self.vals.iter().zip(rhs.vals.iter()).enumerate() {
            vals[i] = *sum.0 / *sum.1;
        }
        Vector { vals }
    }
}

impl<const N: usize, T> Div<T> for Vector<N, T> where T: Value + Div<Output = T> {
    type Output = Vector<N, T>;

    fn div(self, rhs: T) -> Self::Output {
        let mut vals = [T::default(); N];
        vals.iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = self.vals[i] / rhs);
        Vector { vals }
    }
}

impl<const N: usize, T> MulAssign for Vector<N, T> where T: Value {
    fn mul_assign(&mut self, rhs: Self) {
        for (i, val) in rhs.vals.iter().enumerate() {
            self.vals[i] *= *val;
        }
    }
}


pub const fn vec3(x: f32, y: f32, z: f32) -> Vector<3, f32> {
    Vector::<3, f32>::new([x, y, z])
}

pub fn dvec3(vals: [f64; 3]) -> Vector<3, f64> {
    Vector::<3, f64>::new(vals)
}

pub fn ivec3(vals: [i32; 3]) -> Vector<3, i32> {
    Vector::<3, i32>::new(vals)
}

pub fn uvec3(vals: [u32; 3]) -> Vector<3, u32> {
    Vector::<3, u32>::new(vals)
}


pub fn vec2(x: f32, y: f32) -> Vector<2, f32> {
    Vector::<2, f32>::new([x, y])
}

pub fn normalize<const N: usize>(v: Vector<N, f32>) -> Vector<N, f32> {
    let magnitude: f32 = v.vals.iter().map(|&v| v * v).sum();
    v / magnitude.sqrt()
}

pub fn length<const N: usize>(v: Vector<N, f32>) -> f32 {
    v.vals.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

pub fn length2<const N: usize>(v: Vector<N, f32>) -> f32 {
    v.vals.iter().map(|&v| v * v).sum::<f32>()
}

pub fn dot<const N: usize>(a: Vector<N, f32>, b: Vector<N, f32>) -> f32 {
    let mut sum = 0.0;
    for i in 0..N {
        sum += a.vals[i] * b.vals[i];
    }
    sum
    // a.vals.iter().zip(b.vals).map(|(l, r)| l * r).sum::<f32>()
}

pub fn cross(a: Vector<3, f32>, b: Vector<3, f32>) -> Vector<3, f32> {
    vec3(
        a.y() * b.z() - a.z() * b.y(),
        a.z() * b.x() - a.x() * b.z(),
        a.x() * b.y() - a.y() * b.x()
    )
}

pub fn reflect(dir: Vector<3, f32>, normal: Vector<3, f32>) -> Vector<3, f32> {
    dir - 2.0 * dot(dir, normal) * normal
}

pub fn rem(v: Vector<3, f32>, m: Vector<3, f32>) -> Vector<3, f32> {
    vec3(v.x().rem_euclid(m.x()), v.y().rem_euclid(m.y()), v.z().rem_euclid(m.z()))
}

impl<T> Vector<3, T> where T: Default + Copy + Clone {
    pub fn x(&self) -> T {
        self.vals[0]
    }

    pub fn y(&self) -> T {
        self.vals[1]
    }

    pub fn z(&self) -> T {
        self.vals[2]
    }
}

impl Vector<3, f32> {
    pub fn to_rgb_u8(&self) -> [u8; 3] {
        self.vals.map(|f| (f.clamp(0.0, 1.0) * 255.0) as u8)
    }

}

impl<T> Vector<2, T> where T: Default + Copy + Clone {
    pub fn x(&self) -> T {
        self.vals[0]
    }

    pub fn y(&self) -> T {
        self.vals[1]
    }
}


// pub fn dot<const N: usize, T>(a: Vector<N, T>, b: Vector<N, T>, ior: f32) -> Vector<N, T> where T: Default + Clone + Copy + Mul {
//     let vals = a.vals.iter()
//         .zip(b.vals)
//         .map()
//     // Vector::<N, T>::new(a.vals.iter().zip(b.vals.iter()).map(|(l, r)| *l * *r).reduce(|accum, i| accum + i))
// }

// pub fn refract<const N: usize, T>(incident: Vector<N, T>, normal: Vector<N, T>, ior: f32) where T: Default + Clone + Copy {
//     let k: f32 = 1.0 - ior * ior * (1.0 - dot(normal, incident) * dot(normal, incident));
// }

macro_rules! create_vec_constructors {
    ($func_name:ident) => {
        fn $func_name() {
            println!("You called {:?}()", stringify!($func_name));
        }
    }
}

create_vec_constructors!(sdf);