//! Image decoding (model-agnostic; backends receive an RGB8 image).

use backend_api::image::RgbImage;

use crate::error::{ApiError, Result};

/// Decode `bytes` to RGB8, rejecting images above `max_pixels` *before*
/// allocating the pixel buffer (decompression-bomb guard).
///
/// EXIF orientation is intentionally not applied, mirroring Ultralytics
/// (`cv2.imread` semantics) so boxes match its output.
pub fn decode(bytes: &[u8], max_pixels: u64) -> Result<RgbImage> {
    let corrupt = || ApiError::BadRequest("unsupported or corrupt image file".into());
    let reader = || {
        image::ImageReader::new(std::io::Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|_| corrupt())
    };

    let (w, h) = reader()?.into_dimensions().map_err(|_| corrupt())?;
    if w == 0 || h == 0 || (w as u64) * (h as u64) > max_pixels {
        return Err(ApiError::BadRequest(format!(
            "image dimensions {w}x{h} not supported (max {max_pixels} pixels)"
        )));
    }

    let mut reader = reader()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(w);
    limits.max_image_height = Some(h);
    // Worst case 16-bit RGBA (8 B/px) plus decoder scratch.
    limits.max_alloc = Some(max_pixels.saturating_mul(8).saturating_add(64 << 20));
    reader.limits(limits);
    Ok(reader.decode().map_err(|_| corrupt())?.into_rgb8())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(w: u32, h: u32) -> Vec<u8> {
        let mut buf = std::io::Cursor::new(Vec::new());
        RgbImage::new(w, h)
            .write_to(&mut buf, image::ImageFormat::Png)
            .unwrap();
        buf.into_inner()
    }

    #[test]
    fn rejects_oversized_before_decoding() {
        let err = decode(&png(100, 100), 9_999).unwrap_err();
        assert!(err.to_string().contains("100x100"), "{err}");
        assert_eq!(
            decode(&png(100, 100), 10_000).unwrap().dimensions(),
            (100, 100)
        );
    }

    #[test]
    fn rejects_garbage() {
        assert!(decode(b"hello", 1_000_000).is_err());
    }
}
