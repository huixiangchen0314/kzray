//! `kzray-math::fresnel` 的集成测试。

use kzray_math::fresnel::{
    fresnel_conductor, fresnel_dielectric, fresnel_schlick, fresnel_schlick_dielectric, ior,
};
use std::f32::consts::PI;

// ── 介电 Fresnel ──

#[test]
fn dielectric_vertical_incidence_matches_formula() {
    // cos = 1（垂直入射）：F = ((ηi - ηt) / (ηi + ηt))²
    let cases = [
        (1.0, 1.5),     // 空气→玻璃：F0 ≈ 0.04
        (1.0, 1.333),   // 空气→水：F0 ≈ 0.02
        (1.0, 2.417),   // 空气→钻石：F0 ≈ 0.17
    ];
    for (ior_i, ior_t) in cases {
        let f = fresnel_dielectric(1.0, ior_i, ior_t);
        let expected = ((ior_i - ior_t) / (ior_i + ior_t)).powi(2);
        assert!((f - expected).abs() < 1e-6, "({ior_i},{ior_t}) f={f} expected={expected}");
    }
}

#[test]
fn dielectric_grazing_approaches_one() {
    for cos in [0.1f32, 0.05, 0.01, 0.001] {
        let f = fresnel_dielectric(cos, 1.0, 1.5);
        assert!(f > 0.5, "cos={cos} f={f}，掠射应接近 1");
    }
    // cos → 0 时 F → 1
    let f = fresnel_dielectric(1e-6, 1.0, 1.5);
    assert!((f - 1.0).abs() < 1e-3);
}

#[test]
fn dielectric_equal_ior_gives_zero() {
    // ηi = ηt：无界面，无反射
    for cos in [1.0f32, 0.5, 0.1, 0.01] {
        let f = fresnel_dielectric(cos, 1.5, 1.5);
        assert!(f.abs() < 1e-6, "cos={cos} f={f}");
    }
}

#[test]
fn dielectric_tir_from_dense_to_rare() {
    // 玻璃 → 空气，全内反射。临界角 sinθ = 1 / 1.5 → θ ≈ 41.8°
    // cos(临界角) ≈ 0.745
    let cos_critical = (1.0 - (1.0 / 1.5f32).powi(2)).sqrt();   // ≈ 0.745

    // 临界角以内：F < 1
    let f_inside = fresnel_dielectric(0.9, 1.5, 1.0);
    assert!(f_inside < 1.0, "cos=0.9 f={f_inside}");

    // 超过临界角：F = 1（TIR）
    for cos in [0.7f32, 0.5, 0.1, 0.0] {
        assert!(cos < cos_critical);
        let f = fresnel_dielectric(cos, 1.5, 1.0);
        assert!((f - 1.0).abs() < 1e-6, "cos={cos} f={f} 应发生 TIR");
    }
}

#[test]
fn dielectric_negative_cos_swaps_ior() {
    // 从内部出射：cos 为负，内部交换 ior
    let f_positive = fresnel_dielectric(0.5, 1.0, 1.5);
    let f_negative = fresnel_dielectric(-0.5, 1.5, 1.0);
    assert!((f_positive - f_negative).abs() < 1e-6,
        "f_positive={f_positive} f_negative={f_negative}");
}

#[test]
fn dielectric_energy_conservation() {
    // 扫描所有角度，F ∈ [0, 1]
    let iors = [(1.0, 1.5), (1.0, 1.333), (1.5, 1.0), (1.5, 2.4), (1.0, 4.0)];
    for (ior_i, ior_t) in iors {
        for i in 0..=100 {
            let cos = i as f32 / 100.0;
            let f = fresnel_dielectric(cos, ior_i, ior_t);
            assert!(f >= -1e-6 && f <= 1.0 + 1e-6,
                "({ior_i},{ior_t}) cos={cos} f={f} 超出 [0,1]");
            assert!(f.is_finite(), "({ior_i},{ior_t}) cos={cos} f={f} 非有限");
        }
    }
}

#[test]
fn dielectric_energy_conservation_sin_theta_weighted() {
    // 在所有角度上按立体角均匀扫描（sinθ dθ），F 必须是合法的反射率
    let n = 1000;
    for ior_pair in [(1.0, 1.5), (1.5, 2.4), (1.0, 1.333)] {
        let mut max_f = 0.0f32;
        let mut min_f = 1.0f32;
        for i in 0..n {
            let theta = PI * (i as f32 + 0.5) / n as f32;
            let cos = theta.cos();
            if cos < 0.0 { continue; }
            let f = fresnel_dielectric(cos, ior_pair.0, ior_pair.1);
            max_f = max_f.max(f);
            min_f = min_f.min(f);
        }
        assert!(min_f >= -1e-6);
        assert!(max_f <= 1.0 + 1e-6);
    }
}

// ── Schlick 近似 ──

#[test]
fn schlick_at_vertical_equals_f0() {
    for f0 in [0.02f32, 0.04, 0.17, 0.5, 0.9] {
        let f = fresnel_schlick(1.0, f0);
        assert!((f - f0).abs() < 1e-6, "f0={f0} f={f}");
    }
}

#[test]
fn schlick_at_grazing_approaches_one() {
    for f0 in [0.02f32, 0.04, 0.17] {
        let f = fresnel_schlick(0.0, f0);
        assert!((f - 1.0).abs() < 1e-6, "f0={f0} f={f}");
    }
}

#[test]
fn schlick_dielectric_matches_exact_at_vertical() {
    // 垂直入射时 Schlick 与精确介电一致
    for (ior_i, ior_t) in [(1.0, 1.5), (1.0, 1.333), (1.0, 2.4)] {
        let exact = fresnel_dielectric(1.0, ior_i, ior_t);
        let schlick = fresnel_schlick_dielectric(1.0, ior_i, ior_t);
        assert!((exact - schlick).abs() < 1e-6);
    }
}

#[test]
fn schlick_dielectric_underestimates_grazing_for_dense_medium() {
    // Schlick 不含 TIR，掠射时可能不趋于 1（对大 ior 差距）
    // 但必须是合法的反射率（[0, 1]）
    for (ior_i, ior_t) in [(1.0, 1.5), (1.0, 2.4)] {
        for i in 0..=100 {
            let cos = i as f32 / 100.0;
            let f = fresnel_schlick_dielectric(cos, ior_i, ior_t);
            assert!(f >= -1e-6 && f <= 1.0 + 1e-6);
        }
    }
}

// ── 导体 Fresnel ──

#[test]
fn conductor_vertical_incidence_matches_formula() {
    // 垂直入射：F0 = ((η-1)² + k²) / ((η+1)² + k²)
    let cases = [
        (0.15, 3.5),   // 金（红光近似）
        (0.05, 4.0),   // 银
        (1.3, 7.0),    // 铝
        (2.5, 2.5),
    ];
    for (eta, k) in cases {
        let f = fresnel_conductor(1.0, eta, k);
        let expected = ((eta - 1.0).powi(2) + k * k) / ((eta + 1.0).powi(2) + k * k);
        assert!((f - expected).abs() < 1e-4,
            "(eta={eta},k={k}) f={f} expected={expected}");
    }
}

#[test]
fn conductor_grazing_approaches_one() {
    for (eta, k) in [(0.15f32, 3.5f32), (0.05, 4.0), (1.3, 7.0)] {
        let f = fresnel_conductor(1e-6, eta, k);
        assert!((f - 1.0).abs() < 1e-2, "(eta={eta},k={k}) 掠射 f={f}");
    }
}

#[test]
fn conductor_energy_conservation() {
    for (eta, k) in [(0.15f32, 3.5f32), (0.05, 4.0), (1.3, 7.0), (0.5, 1.0)] {
        for i in 0..=100 {
            let cos = i as f32 / 100.0;
            let f = fresnel_conductor(cos, eta, k);
            assert!(f >= -1e-6 && f <= 1.0 + 1e-6,
                "(eta={eta},k={k}) cos={cos} f={f}");
            assert!(f.is_finite());
        }
    }
}

// 导体非偏振平均反射率 F=(Rs+Rp)/2 对 cosθ 不一定单调。
// Rp 在 pseudo-Brewster 角有非零极小；当 eta<1 且 k 较大时，
// 这可使 F 出现很浅的极小值（dip），而不是严格随角度增大而增大。
// 金/银在可见光常见 eta<1；铝的 eta 可 >1，需按波长取参数。
// #[test]
// fn conductor_is_monotonic_in_cos() {
//     // 随着 cos 减小（角度增大），Fresnel 应单调增大
//     let (eta, k) = (0.15f32, 3.5f32);
//     let mut prev = 0.0f32;
//     let mut cos = 1.0f32;
//     while cos >= 0.0 {
//         let f = fresnel_conductor(cos, eta, k);
//         assert!(f >= prev - 1e-5, "cos={cos} f={f} 逆序 prev={prev}");
//         prev = f;
//         cos -= 0.05;
//     }
// }

#[test]
fn conductor_fresnel_grows_toward_grazing() {
    // 普适规律：掠射趋于 1，且是全局最大值。
    // 不对"45° > 垂直"或"近掠射 > 45°"做断言：
    // 良导体（尤其铝 eta=1.3,k=7）垂直入射已 >0.90，
    // 45° 附近非偏振平均 F 因 Rp 的 pseudo-Brewster 凹陷可能略低，
    // 且凹陷可延伸到 cos≈0.05 仍低于 45°。这是真实物理，非数值误差。
    for (eta, k) in [(0.15f32, 3.5f32), (0.05, 4.0), (1.3, 7.0)] {
        let f_vertical = fresnel_conductor(1.0, eta, k);
        let f_grazing = fresnel_conductor(0.0, eta, k);

        // 垂直入射满足闭式公式
        let expected = ((eta - 1.0).powi(2) + k * k)
            / ((eta + 1.0).powi(2) + k * k);
        assert!(
            (f_vertical - expected).abs() < 1e-5,
            "eta={eta},k={k}: 垂直入射 {f_vertical} vs 公式 {expected}"
        );

        // 掠射趋于 1
        assert!(
            (f_grazing - 1.0).abs() < 1e-3,
            "eta={eta},k={k}: 掠射 {f_grazing} 应趋于 1"
        );

        // 掠射是全局最大值，且 F ∈ [0,1]
        for i in 0..=32 {
            let c = i as f32 / 32.0;
            let f = fresnel_conductor(c, eta, k);
            assert!(
                (0.0..=1.0).contains(&f),
                "eta={eta},k={k}, cos={c}: F={f} 超出 [0,1]"
            );
            assert!(
                f <= f_grazing + 1e-6,
                "eta={eta},k={k}, cos={c}: F={f} 超过掠射值 {f_grazing}"
            );
        }
    }
}

#[test]
fn conductor_near_vertical_dip_for_eta_lt_one() {
    // eta<1 的金属（金、银）：45° 处 F 略高于垂直入射
    // （Rp 凹陷不足以抵消 Rs 上升）。铝 eta>1 不适用。
    for (eta, k) in [(0.15f32, 3.5f32), (0.05, 4.0)] {
        let f_vertical = fresnel_conductor(1.0, eta, k);
        let f_45 = fresnel_conductor(std::f32::consts::FRAC_1_SQRT_2, eta, k);
        assert!(
            f_45 > f_vertical,
            "eta={eta},k={k}: 45° ({f_45}) 应 > 垂直 ({f_vertical})"
        );
    }
}

#[test]
fn conductor_zero_k_approaches_dielectric() {
    // k = 0 的导体退化为"完美介质导体"（如理想金属的反极限）
    // 垂直入射公式 ((η-1)/(η+1))²，与介电 ior_i=1, ior_t=η 一致
    for eta in [1.5f32, 2.0, 3.0] {
        let f_conductor = fresnel_conductor(1.0, eta, 0.0);
        let f_dielectric = fresnel_dielectric(1.0, 1.0, eta);
        assert!((f_conductor - f_dielectric).abs() < 1e-4,
            "eta={eta} conductor={f_conductor} dielectric={f_dielectric}");
    }
}

// ── IOR 常量 ──

#[test]
fn ior_constants_sane() {
    assert!((ior::AIR - 1.0).abs() < 1e-6);
    assert!(ior::WATER > 1.3 && ior::WATER < 1.4);
    assert!(ior::GLASS > 1.4 && ior::GLASS < 1.6);
    assert!(ior::DIAMOND > 2.0 && ior::DIAMOND < 3.0);
}
