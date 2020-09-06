use num_traits::Num;
use std::clone::Clone;

pub trait Numeric: 
    Num
    + Clone
{
}

impl Numeric for i32 {}
impl Numeric for i64 {}
impl Numeric for f32 {}
impl Numeric for f64 {}
