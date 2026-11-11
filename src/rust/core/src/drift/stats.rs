/// Calculate the Chi-squared contingency statistic for a 2x2 table.
///
/// The table is represented as:
/// [[a, b],
///  [c, d]]
///
/// Formula: (a+b+c+d) * (ad - bc)^2 / ((a+b)(c+d)(a+c)(b+d))
pub fn chi2_contingency_2x2(a: f64, b: f64, c: f64, d: f64) -> f64 {
    let n = a + b + c + d;
    if n == 0.0 {
        return 0.0;
    }
    let numerator = n * (a * d - b * c).powi(2);
    let denominator = (a + b) * (c + d) * (a + c) * (b + d);
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chi2_contingency() {
        // Example case
        let stat = chi2_contingency_2x2(10.0, 5.0, 10.0, 20.0);
        // Table: [[10, 5], [10, 20]]
        // n = 45
        // (10*20 - 5*10)^2 * 45 / (15 * 30 * 20 * 25)
        // 150^2 * 45 / (225000) = 22500 * 45 / 225000 = 4.5
        assert!((stat - 4.5).abs() < 1e-7);
    }
}
