//! 确定性随机数生成。每线程独立，不共享。

/// 采样器。`dimension` 是采样维度——PCG 忽略，Sobol 用它。
///
/// 构造时绑定 `(seed, sample_index)`，`seed` 由 `seed_for` 派生。
/// **只 `Send` 不 `Sync`**——每个线程自己的 Sampler，不做共享。
pub trait Sampler: Send {
    fn next_f32(&mut self, dimension: u32) -> f32;

    fn next_2d(&mut self, dimension: u32) -> (f32, f32) {
        (self.next_f32(dimension * 2), self.next_f32(dimension * 2 + 1))
    }

    fn next_array(&mut self, dimension: u32, out: &mut [f32]) {
        for (i, v) in out.iter_mut().enumerate() {
            *v = self.next_f32(dimension + i as u32);
        }
    }
}
