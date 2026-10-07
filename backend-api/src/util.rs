//! Small helpers shared by backends (tensor layout, output lookup, decode
//! finishing). Nothing here is model-specific.

use image::RgbImage;

use crate::{BackendError, Detection, OutputTensor, Result};

/// Find an output by name, falling back to its position in the response.
pub fn output<'a>(
    outputs: &'a [OutputTensor],
    name: &str,
    index: usize,
) -> Result<&'a OutputTensor> {
    outputs
        .iter()
        .find(|o| o.name == name)
        .or_else(|| outputs.get(index))
        .ok_or_else(|| BackendError::BadOutput(format!("triton returned no '{name}' output")))
}

/// Per-image dims of an output whose shape may or may not carry a leading
/// batch dim of 1 (Triton includes it when `max_batch_size > 0`).
pub fn per_image_dims(t: &OutputTensor, rank: usize) -> Result<&[u64]> {
    match t.shape.len() {
        n if n == rank => Ok(&t.shape),
        n if n == rank + 1 && t.shape[0] == 1 => Ok(&t.shape[1..]),
        _ => Err(BackendError::BadOutput(format!(
            "unexpected '{}' shape {:?} (expected rank {rank} per image)",
            t.name, t.shape
        ))),
    }
}

/// Ensure an output carries at least `n` values.
pub fn require_len(t: &OutputTensor, n: usize) -> Result<()> {
    if t.data.len() < n {
        return Err(BackendError::BadOutput(format!(
            "'{}' output too small: {} values, expected {n}",
            t.name,
            t.data.len()
        )));
    }
    Ok(())
}

/// Write an RGB image into a planar CHW f32 buffer of a `width x height`
/// canvas at offset `(left, top)`, applying `f(channel, value)` per sample.
/// The rest of `out` is left untouched (callers pre-fill padding).
pub fn write_chw(
    img: &RgbImage,
    out: &mut [f32],
    width: u32,
    height: u32,
    left: u32,
    top: u32,
    f: impl Fn(usize, u8) -> f32,
) {
    let (w, h) = (width as usize, height as usize);
    let plane = w * h;
    let (iw, ih) = (img.width() as usize, img.height() as usize);
    let (left, top) = (left as usize, top as usize);
    let raw = img.as_raw();
    for y in 0..ih.min(h.saturating_sub(top)) {
        let row = &raw[y * iw * 3..(y + 1) * iw * 3];
        let base = (top + y) * w + left;
        for x in 0..iw.min(w.saturating_sub(left)) {
            let px = &row[x * 3..x * 3 + 3];
            for c in 0..3 {
                out[c * plane + base + x] = f(c, px[c]);
            }
        }
    }
}

/// Numerically stable logistic sigmoid.
pub fn sigmoid(x: f64) -> f64 {
    let x = x.clamp(-88.0, 88.0);
    1.0 / (1.0 + (-x).exp())
}

/// Clamp a box to the original image bounds.
pub fn clamped(class: i64, confidence: f64, b: (f64, f64, f64, f64), w: u32, h: u32) -> Detection {
    let (w, h) = (w as f64, h as f64);
    Detection {
        class,
        confidence,
        x1: b.0.clamp(0.0, w),
        y1: b.1.clamp(0.0, h),
        x2: b.2.clamp(0.0, w),
        y2: b.3.clamp(0.0, h),
    }
}

/// Sort by descending confidence and cap the count.
pub fn finish(mut dets: Vec<Detection>, max_det: usize) -> Vec<Detection> {
    dets.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    dets.truncate(max_det);
    dets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_chw_places_pixels_with_offset() {
        let img = RgbImage::from_fn(2, 1, |x, _| image::Rgb([x as u8 + 1, 10, 20]));
        let mut out = vec![0f32; 3 * 4 * 2];
        write_chw(&img, &mut out, 4, 2, 1, 1, |_, v| v as f32);
        // channel 0, row 1, cols 1..3
        assert_eq!(&out[4..8], &[0.0, 1.0, 2.0, 0.0]);
        // channel 2 plane starts at 16
        assert_eq!(&out[16 + 4..16 + 8], &[0.0, 20.0, 20.0, 0.0]);
    }

    #[test]
    fn per_image_dims_accepts_batched_and_unbatched() {
        let t = |shape: Vec<u64>| OutputTensor {
            name: "x".into(),
            shape,
            data: vec![],
        };
        assert_eq!(per_image_dims(&t(vec![1, 300, 6]), 2).unwrap(), &[300, 6]);
        assert_eq!(per_image_dims(&t(vec![300, 6]), 2).unwrap(), &[300, 6]);
        assert!(per_image_dims(&t(vec![2, 300, 6]), 2).is_err());
    }
}
