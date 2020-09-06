// use itertools::Itertools;
use ndarray::prelude::*;
// use ndarray::{LinalgScalar, NdFloat};
// use num_traits::Num;

use crate::numeric_traits::Numeric;

pub fn chi_merge<T: Numeric>(feature: ArrayView1<T>, target: ArrayView1<T>) -> Array1<T> {
    &feature + &target
}


fn c_unique<T: Numeric>(feature: Array1<T>) -> Vec<T> {
    let mut v = feature.to_vec();
    v.dedup();
    v
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    // #[ignore]
    fn test_chi_merge() {
        let feat = Array::range(0., 500., 1.);
        let target = Array::ones(500);

        let res = chi_merge(feat.view(), target.view());
        println!("{}", res);
    }

    #[test]
    // #[ignore]
    fn test_unique() {
        let target: Array1<f64> = Array::ones(500);

        let res = c_unique(target);
        println!("{:?}", res);
    }
}