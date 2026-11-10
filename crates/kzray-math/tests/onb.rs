//! `kzray-math::onb` 的集成测试。

use glam::Vec3;
use kzray_math::onb::Onb;

/// 球面均匀采样方向（用于随机法线测试）。
fn random_dir(i: usize) -> Vec3 {
    let u1 = (i as f32 * 0.618_033_988_7).fract();
    let u2 = (i as f32 * 0.754_877_666_2).fract();
    let theta = (2.0 * u1 - 1.0).acos();
    let phi = 2.0 * std::f32::consts::PI * u2;
    Vec3::new(
        theta.sin() * phi.cos(),
        theta.sin() * phi.sin(),
        theta.cos(),
    )
}

#[test]
fn identity_normal_gives_identity_basis() {
    let onb = Onb::from_w(Vec3::Z);
    assert!((onb.w - Vec3::Z).length() < 1e-6);
    assert!((onb.u - Vec3::X).length() < 1e-6);
    assert!((onb.v - Vec3::Y).length() < 1e-6);
    assert!((onb.u.cross(onb.v) - onb.w).length() < 1e-6);
}

#[test]
fn negative_z_is_robust() {
    // Duff 2017 修正的经典退化方向
    let onb = Onb::from_w(-Vec3::Z);
    assert!((onb.w - -Vec3::Z).length() < 1e-6);
    assert!((onb.u.length() - 1.0).abs() < 1e-5);
    assert!((onb.v.length() - 1.0).abs() < 1e-5);
    assert!(onb.u.dot(onb.v).abs() < 1e-5);
    assert!(onb.u.dot(onb.w).abs() < 1e-5);
    assert!((onb.u.cross(onb.v) - onb.w).length() < 1e-5);
}

#[test]
fn random_normals_produce_orthonormal_basis() {
    for i in 0..1000 {
        let n = random_dir(i);
        let onb = Onb::from_w(n);

        // 正交性
        assert!((onb.u.dot(onb.v)).abs() < 1e-5, "i={i} u·v={}", onb.u.dot(onb.v));
        assert!((onb.u.dot(onb.w)).abs() < 1e-5, "i={i} u·w={}", onb.u.dot(onb.w));
        assert!((onb.v.dot(onb.w)).abs() < 1e-5, "i={i} v·w={}", onb.v.dot(onb.w));

        // 单位长
        assert!((onb.u.length() - 1.0).abs() < 1e-4, "i={i} |u|={}", onb.u.length());
        assert!((onb.v.length() - 1.0).abs() < 1e-4, "i={i} |v|={}", onb.v.length());
        assert!((onb.w.length() - 1.0).abs() < 1e-4, "i={i} |w|={}", onb.w.length());

        // 右手系
        assert!((onb.u.cross(onb.v) - onb.w).length() < 1e-4, "i={i}");
    }
}

#[test]
fn to_world_to_local_roundtrip() {
    let onb = Onb::from_w(Vec3::new(0.3, 0.5, 0.8).normalize());
    for local in [
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.5, -0.3, 0.7).normalize(),
        Vec3::new(-0.2, 0.4, -0.9).normalize(),
    ] {
        let world = onb.to_world(local);
        let back = onb.to_local(world);
        assert!((back - local).length() < 1e-5, "local={local:?} back={back:?}");
    }
}

#[test]
fn to_world_preserves_length() {
    let onb = Onb::from_w(Vec3::new(0.3, 0.5, 0.8).normalize());
    for local in [
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.5, -0.3, 0.7).normalize(),
    ] {
        let world = onb.to_world(local);
        assert!((world.length() - local.length()).abs() < 1e-5);
    }
}

#[test]
fn w_axis_maps_to_normal() {
    let n = Vec3::new(0.6, -0.8, 0.2).normalize();
    let onb = Onb::from_w(n);
    let world = onb.to_world(Vec3::new(0.0, 0.0, 1.0));
    assert!((world - n).length() < 1e-5);
}

#[test]
fn local_z_is_one_when_world_equals_normal() {
    let n = Vec3::new(-0.3, 0.7, 0.5).normalize();
    let onb = Onb::from_w(n);
    let local = onb.to_local(n);
    assert!(local.x.abs() < 1e-5);
    assert!(local.y.abs() < 1e-5);
    assert!((local.z - 1.0).abs() < 1e-5);
}

#[test]
fn axis_aligned_normals() {
    for n in [
        Vec3::X, -Vec3::X,
        Vec3::Y, -Vec3::Y,
        Vec3::Z, -Vec3::Z,
    ] {
        let onb = Onb::from_w(n);
        assert!((onb.u.length() - 1.0).abs() < 1e-5, "n={n:?}");
        assert!((onb.v.length() - 1.0).abs() < 1e-5, "n={n:?}");
        assert!(onb.u.dot(onb.v).abs() < 1e-5, "n={n:?}");
        assert!(onb.u.dot(onb.w).abs() < 1e-5, "n={n:?}");
        assert!((onb.u.cross(onb.v) - onb.w).length() < 1e-5, "n={n:?}");
    }
}

#[test]
fn near_negative_z_robust() {
    // n.z ≈ -1 附近的浮点边界
    for dz in [1e-6f32, 1e-5, 1e-4, 1e-3] {
        let n = Vec3::new(dz, dz, -(1.0 - 2.0 * dz * dz).sqrt()).normalize();
        let onb = Onb::from_w(n);
        assert!((onb.u.length() - 1.0).abs() < 1e-4, "dz={dz}");
        assert!((onb.v.length() - 1.0).abs() < 1e-4, "dz={dz}");
        assert!(onb.u.dot(onb.v).abs() < 1e-4, "dz={dz}");
    }
}

#[test]
fn hemisphere_sampling_stays_in_hemisphere() {
    // 局部半球 (x, y, z) with z ≥ 0 采样后，转到世界空间，
    // 所有方向 dot(n) ≥ 0（因为局部 z 分量非负）
    let n = Vec3::new(0.3, 0.5, 0.8).normalize();
    let onb = Onb::from_w(n);
    for i in 0..1000 {
        let u1 = (i as f32 * 0.618_033_988_7).fract();
        let u2 = (i as f32 * 0.754_877_666_2).fract();
        let r = u1.sqrt();
        let phi = 2.0 * std::f32::consts::PI * u2;
        let local = Vec3::new(r * phi.cos(), r * phi.sin(), (1.0 - u1).sqrt());
        let world = onb.to_world(local);
        assert!(world.dot(n) >= -1e-5, "i={i} dot={}", world.dot(n));
    }
}
