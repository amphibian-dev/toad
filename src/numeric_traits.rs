use num_traits::{Num};
use std::clone::Clone;
use std::cmp::PartialOrd;

pub trait Numeric: 
    Num
    + Clone
    + PartialOrd
{}

impl Numeric for i32 {}
impl Numeric for i64 {}
impl Numeric for f32 {}
impl Numeric for f64 {}
