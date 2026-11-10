//! `kzray-math::dist` 的集成测试。
//! 测试中，使用黄金比构造低差异序列

use kzray_math::dist::{Distribution1D, Distribution2D};

// ── Distribution1D：构造与不变量 ──

#[test]
fn uniform_weights_sum() {
    let d = Distribution1D::new(&[1.0, 1.0, 1.0, 1.0]);
    assert_eq!(d.len(), 4);
    assert!((d.sum() - 4.0).abs() < 1e-6);
    assert!((d.cdf()[0]).abs() < 1e-12);
    assert!((d.cdf()[4] - 4.0).abs() < 1e-6);
}

#[test]
fn single_element() {
    let d = Distribution1D::new(&[5.0]);
    assert_eq!(d.len(), 1);
    assert!((d.sum() - 5.0).abs() < 1e-6);
    let (idx, pdf) = d.sample_discrete(0.5);
    assert_eq!(idx, 0);
    assert!((pdf - 1.0).abs() < 1e-6);
}

#[test]
#[should_panic]
fn empty_panics() {
    Distribution1D::new(&[]);
}

#[test]
#[should_panic]
fn all_zero_panics() {
    Distribution1D::new(&[0.0, 0.0, 0.0]);
}

#[test]
#[should_panic]
fn negative_panics() {
    Distribution1D::new(&[1.0, -1.0, 2.0]);
}

#[test]
#[should_panic]
fn nan_panics() {
    Distribution1D::new(&[1.0, f32::NAN, 1.0]);
}

#[test]
#[should_panic]
fn inf_panics() {
    Distribution1D::new(&[1.0, f32::INFINITY, 1.0]);
}

// ── Distribution1D：离散采样 ──

#[test]
fn uniform_discrete_sampling() {
    let d = Distribution1D::new(&[1.0, 1.0, 1.0, 1.0]);
    // u 在四个等分区间：0.0, 0.25, 0.5, 0.75 分别对应 0,1,2,3
    assert_eq!(d.sample_discrete(0.0).0, 0);
    assert_eq!(d.sample_discrete(0.24).0, 0);
    assert_eq!(d.sample_discrete(0.26).0, 1);
    assert_eq!(d.sample_discrete(0.49).0, 1);
    assert_eq!(d.sample_discrete(0.51).0, 2);
    assert_eq!(d.sample_discrete(0.74).0, 2);
    assert_eq!(d.sample_discrete(0.76).0, 3);
    assert_eq!(d.sample_discrete(0.99).0, 3);

    for i in 0..4 {
        assert!((d.pdf_discrete(i) - 0.25).abs() < 1e-6);
    }
}

#[test]
fn nonuniform_discrete_frequencies() {
    // weights [1, 3]：index 0 约 25%，index 1 约 75%
    let d = Distribution1D::new(&[1.0, 3.0]);
    let n = 100_000;
    let mut counts = [0u32; 2];
    for i in 0..n {
        let u = (i as f32 + 0.5) / n as f32;
        let (idx, _) = d.sample_discrete(u);
        counts[idx] += 1;
    }
    let p0 = counts[0] as f32 / n as f32;
    let p1 = counts[1] as f32 / n as f32;
    assert!((p0 - 0.25).abs() < 0.01, "p0={p0}");
    assert!((p1 - 0.75).abs() < 0.01, "p1={p1}");
}

#[test]
fn sample_discrete_pdf_consistent() {
    let d = Distribution1D::new(&[2.0, 3.0, 1.0, 4.0]);
    let n = 1000;
    for i in 0..n {
        let u = (i as f32 + 0.5) / n as f32;
        let (idx, pdf) = d.sample_discrete(u);
        let expected = d.pdf_discrete(idx);
        assert!(
            (pdf - expected).abs() < 1e-6,
            "u={u} idx={idx} pdf={pdf} expected={expected}"
        );
    }
}

#[test]
fn u_one_does_not_panic() {
    let d = Distribution1D::new(&[1.0, 2.0, 3.0]);
    let (idx, _) = d.sample_discrete(1.0);
    assert!(idx < 3);
    let (x, _) = d.sample_continuous(1.0);
    assert!(x < 3.0);
}

#[test]
fn u_zero_returns_first() {
    let d = Distribution1D::new(&[1.0, 2.0, 3.0]);
    assert_eq!(d.sample_discrete(0.0).0, 0);
    let (x, _) = d.sample_continuous(0.0);
    assert!(x.abs() < 1e-6);
}

// ── Distribution1D：连续采样 ──

#[test]
fn continuous_uniform_range() {
    let d = Distribution1D::new(&[1.0, 1.0, 1.0, 1.0]);
    for i in 0..100 {
        let u = i as f32 / 100.0;
        let (x, _) = d.sample_continuous(u);
        assert!((0.0..4.0).contains(&x), "x={x} 超出 [0, 4)");
    }
}

#[test]
fn continuous_uniform_pdf_is_constant() {
    let d = Distribution1D::new(&[1.0, 1.0, 1.0, 1.0]);
    // 均匀权重下 p(x) = 1/4
    for i in 0..10 {
        let x = i as f32 * 0.4;
        let pdf = d.pdf_continuous(x);
        assert!((pdf - 0.25).abs() < 1e-6, "x={x} pdf={pdf}");
    }
}

#[test]
fn continuous_linear_interpolation_within_bin() {
    // weights [1, 3]：cdf = [0, 1, 4]
    // u 使 target ∈ [0, 1) 时，x = target / 1（即 idx=0 的 frac）
    // u 使 target ∈ [1, 4) 时，x = 1 + (target - 1) / 3
    let d = Distribution1D::new(&[1.0, 3.0]);
    let (x, pdf) = d.sample_continuous(0.5);   // target = 0.5 * 4 = 2
    // target=2 落在 [1, 4)，frac = (2-1)/3 = 1/3，x = 1 + 1/3
    assert!((x - (1.0 + 1.0 / 3.0)).abs() < 1e-5, "x={x}");
    assert!((pdf - 3.0 / 4.0).abs() < 1e-5, "pdf={pdf}");   // w[1]/sum = 3/4
}

#[test]
fn pdf_continuous_outside_range_is_zero() {
    let d = Distribution1D::new(&[1.0, 1.0]);
    assert_eq!(d.pdf_continuous(-0.1), 0.0);
    assert_eq!(d.pdf_continuous(2.0), 0.0);
    assert_eq!(d.pdf_continuous(2.5), 0.0);
}

#[test]
fn continuous_pdf_integrates_to_one() {
    // 数值积分 ∫_0^n pdf_continuous(x) dx 应 = 1
    let d = Distribution1D::new(&[2.0, 1.0, 4.0, 1.0]);
    let n = 4.0;
    let samples = 10000;
    let dx = n / samples as f32;
    let mut sum = 0.0f32;
    for i in 0..samples {
        let x = (i as f32 + 0.5) * dx;
        sum += d.pdf_continuous(x) * dx;
    }
    assert!((sum - 1.0).abs() < 1e-3, "∫pdf = {sum}");
}

// ── Distribution1D：from_raw / into_raw ──

#[test]
fn raw_roundtrip_1d() {
    let d = Distribution1D::new(&[1.0, 2.0, 3.0, 4.0]);
    let (w, cdf, sum) = d.clone().into_raw();
    let d2 = Distribution1D::from_raw(w, cdf, sum);
    for i in 0..1000 {
        let u = (i as f32 + 0.5) / 1000.0;
        assert_eq!(d.sample_discrete(u).0, d2.sample_discrete(u).0);
    }
}

#[test]
#[should_panic]
fn from_raw_rejects_bad_cdf_length() {
    Distribution1D::from_raw(vec![1.0, 1.0], vec![0.0, 1.0], 2.0);
}

#[test]
#[should_panic]
fn from_raw_rejects_nonzero_cdf_start() {
    Distribution1D::from_raw(vec![1.0], vec![0.5, 1.0], 1.0);
}

// ── Distribution2D：构造 ──

#[test]
fn uniform_2d_shape() {
    let d = Distribution2D::new(&[
        vec![1.0, 1.0, 1.0],
        vec![1.0, 1.0, 1.0],
    ]);
    assert_eq!(d.rows(), 2);
    assert_eq!(d.cols(), 3);
    assert!((d.total() - 6.0).abs() < 1e-6);
}

#[test]
#[should_panic]
fn empty_2d_panics() {
    Distribution2D::new(&[]);
}

#[test]
#[should_panic]
fn empty_row_2d_panics() {
    Distribution2D::new(&[vec![]]);
}

#[test]
#[should_panic]
fn inconsistent_row_lengths_panics() {
    Distribution2D::new(&[vec![1.0, 1.0], vec![1.0, 1.0, 1.0]]);
}

// ── Distribution2D：离散采样 ──

#[test]
fn uniform_2d_discrete() {
    let d = Distribution2D::new(&[
        vec![1.0, 1.0],
        vec![1.0, 1.0],
    ]);
    for i in 0..2 {
        for j in 0..2 {
            let expected = 0.25;
            assert!((d.pdf_discrete(i, j) - expected).abs() < 1e-6);
        }
    }
}

#[test]
fn nonuniform_2d_frequencies() {
    let d = Distribution2D::new(&[
        vec![1.0, 3.0],
        vec![2.0, 2.0],
    ]);
    let n = 100_000;
    let mut counts = [[0u32; 2]; 2];
    for i in 0..n {
        let u1 = (i as f32 + 0.5) / n as f32;
        // 黄金比 fract：`i * φ mod 1` 在 [0,1) 上均匀分布，且与 u1 独立
        let u2 = (i as f64 * 0.618_033_988_7).fract() as f32;
        let ((r, c), _) = d.sample_discrete(u1, u2);
        counts[r][c] += 1;
    }
    let expected = [[0.125, 0.375], [0.25, 0.25]];
    for r in 0..2 {
        for c in 0..2 {
            let p = counts[r][c] as f32 / n as f32;
            assert!(
                (p - expected[r][c]).abs() < 0.02,
                "({r},{c}) p={p} 期望={}", expected[r][c]
            );
        }
    }
}

#[test]
fn sample_discrete_pdf_consistent_2d() {
    let d = Distribution2D::new(&[
        vec![2.0, 1.0, 3.0],
        vec![1.0, 4.0, 1.0],
    ]);
    let n = 500;
    for i in 0..n {
        let u1 = (i as f32 + 0.5) / n as f32;
        let u2 = (i as f32 * 0.618_033_988_7).fract();
        let ((r, c), pdf) = d.sample_discrete(u1, u2);
        let expected = d.pdf_discrete(r, c);
        assert!((pdf - expected).abs() < 1e-6, "({r},{c}) pdf={pdf} expected={expected}");
    }
}
// ── Distribution2D：连续采样 ──

#[test]
fn continuous_2d_range() {
    let d = Distribution2D::new(&[
        vec![1.0, 2.0],
        vec![3.0, 4.0],
        vec![1.0, 1.0],
    ]);
    for i in 0..100 {
        let u1 = i as f32 / 100.0;
        let u2 = ((i * 37) % 100) as f32 / 100.0;
        let ((u, v), pdf) = d.sample_continuous(u1, u2);
        assert!((0.0..1.0).contains(&u), "u={u}");
        assert!((0.0..1.0).contains(&v), "v={v}");
        assert!(pdf > 0.0 && pdf.is_finite(), "pdf={pdf}");
    }
}

#[test]
fn continuous_2d_pdf_consistent_with_sample() {
    let d = Distribution2D::new(&[
        vec![1.0, 2.0],
        vec![3.0, 4.0],
    ]);
    let n = 200;
    for i in 0..n {
        let u1 = (i as f32 + 0.5) / n as f32;
        let u2 = (i as f64 * 0.618_033_988_7).fract() as f32; // 黄金比 f32存不下
        let ((u, v), pdf_sample) = d.sample_continuous(u1, u2);
        let pdf_lookup = d.pdf_continuous(u, v);
        assert!(
            (pdf_sample - pdf_lookup).abs() < 1e-4,
            "sample pdf={pdf_sample}, lookup pdf={pdf_lookup}, (u,v)=({u},{v})"
        );
    }
}

#[test]
fn continuous_2d_pdf_integrates_to_one() {
    let d = Distribution2D::new(&[
        vec![1.0, 2.0],
        vec![3.0, 1.0],
    ]);
    let n = 200;
    let du = 1.0 / n as f32;
    let mut sum = 0.0f32;
    for i in 0..n {
        for j in 0..n {
            let u = (i as f32 + 0.5) * du;
            let v = (j as f32 + 0.5) * du;
            sum += d.pdf_continuous(u, v) * du * du;
        }
    }
    assert!((sum - 1.0).abs() < 0.05, "∫∫pdf = {sum}");
}

// ── Distribution2D：from_raw / into_raw ──

#[test]
fn raw_roundtrip_2d() {
    let d = Distribution2D::new(&[
        vec![1.0, 2.0],
        vec![3.0, 1.0],
    ]);
    let (rows, marginal) = d.clone().into_raw();
    let d2 = Distribution2D::from_raw(rows, marginal);

    for i in 0..100 {
        let u1 = (i as f32 + 0.5) / 100.0;
        let u2 = (i as f64 * 0.618_033_988_7).fract() as f32;
        let ((r1, c1), p1) = d.sample_discrete(u1, u2);
        let ((r2, c2), p2) = d2.sample_discrete(u1, u2);
        assert_eq!((r1, c1), (r2, c2));
        assert!((p1 - p2).abs() < 1e-6);
    }
}

#[test]
#[should_panic]
fn from_raw_2d_rejects_mismatched_marginal() {
    let rows = vec![
        Distribution1D::new(&[1.0, 1.0]),
        Distribution1D::new(&[1.0, 1.0]),
    ];
    let marginal = Distribution1D::new(&[2.0]);   // 长度 1，行数 2
    Distribution2D::from_raw(rows, marginal);
}
