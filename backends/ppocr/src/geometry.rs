//! Plane geometry for DB text detection and crop rectification, replacing
//! the OpenCV calls of the PaddleOCR reference (`findContours`,
//! `minAreaRect`, `getPerspectiveTransform` + `warpPerspective`).

use backend_api::image::{Rgb, RgbImage};

pub type Point = (f64, f64);

/// 8-connected components of `mask` (`w x h`, row-major), in scan order of
/// their first pixel. Each component is returned as its row extremes (the
/// leftmost and rightmost pixel of every row it touches), which carry the
/// same convex hull as the full component.
pub fn components(mask: &[bool], w: usize, h: usize, max: usize) -> Vec<Vec<Point>> {
    let mut label = vec![false; mask.len()];
    let mut out = Vec::new();
    let mut stack = Vec::new();
    for start in 0..mask.len() {
        if !mask[start] || label[start] {
            continue;
        }
        if out.len() == max {
            break;
        }
        // Per-row (min x, max x) of this component.
        let mut rows: Vec<(usize, usize, usize)> = Vec::new();
        label[start] = true;
        stack.push(start);
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            match rows.iter_mut().find(|r| r.0 == y) {
                Some(r) => {
                    r.1 = r.1.min(x);
                    r.2 = r.2.max(x);
                }
                None => rows.push((y, x, x)),
            }
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if mask[j] && !label[j] {
                        label[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        out.push(
            rows.iter()
                .flat_map(|&(y, x0, x1)| [(x0 as f64, y as f64), (x1 as f64, y as f64)])
                .collect(),
        );
    }
    out
}

fn cross(o: Point, a: Point, b: Point) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

/// Convex hull (Andrew's monotone chain), counter-clockwise, no repeats.
pub fn convex_hull(points: &[Point]) -> Vec<Point> {
    let mut p = points.to_vec();
    p.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    p.dedup();
    if p.len() < 3 {
        return p;
    }
    let mut hull: Vec<Point> = Vec::with_capacity(p.len() * 2);
    for pass in 0..2 {
        let start = hull.len();
        let iter: Box<dyn Iterator<Item = &Point>> = if pass == 0 {
            Box::new(p.iter())
        } else {
            Box::new(p.iter().rev())
        };
        for &pt in iter {
            while hull.len() >= start + 2
                && cross(hull[hull.len() - 2], hull[hull.len() - 1], pt) <= 0.0
            {
                hull.pop();
            }
            hull.push(pt);
        }
        hull.pop();
    }
    hull
}

/// A rotated rectangle: corners ordered top-left, top-right, bottom-right,
/// bottom-left (PaddleOCR `get_mini_boxes` order) and its shorter side.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub corners: [Point; 4],
    pub short_side: f64,
}

/// Minimum-area enclosing rectangle (rotating calipers over hull edges).
/// `None` for degenerate (collinear) inputs.
pub fn min_area_rect(points: &[Point]) -> Option<Rect> {
    let hull = convex_hull(points);
    if hull.len() < 3 {
        return None;
    }
    let mut best: Option<(f64, [Point; 4], f64)> = None;
    for i in 0..hull.len() {
        let (a, b) = (hull[i], hull[(i + 1) % hull.len()]);
        let len = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
        if len == 0.0 {
            continue;
        }
        let u = ((b.0 - a.0) / len, (b.1 - a.1) / len);
        let v = (-u.1, u.0);
        let (mut u0, mut u1, mut v0, mut v1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for p in &hull {
            let (pu, pv) = (p.0 * u.0 + p.1 * u.1, p.0 * v.0 + p.1 * v.1);
            (u0, u1, v0, v1) = (u0.min(pu), u1.max(pu), v0.min(pv), v1.max(pv));
        }
        let area = (u1 - u0) * (v1 - v0);
        if best.as_ref().is_none_or(|b| area < b.0) {
            let at = |s: f64, t: f64| (u.0 * s + v.0 * t, u.1 * s + v.1 * t);
            let corners = [at(u0, v0), at(u1, v0), at(u1, v1), at(u0, v1)];
            best = Some((area, corners, (u1 - u0).min(v1 - v0)));
        }
    }
    let (_, corners, short_side) = best?;
    Some(Rect {
        corners: order_corners(corners),
        short_side,
    })
}

/// PaddleOCR `get_mini_boxes` ordering: of the two leftmost corners the
/// upper is top-left, of the two rightmost the upper is top-right.
pub fn order_corners(mut c: [Point; 4]) -> [Point; 4] {
    c.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let (tl, bl) = if c[1].1 > c[0].1 {
        (c[0], c[1])
    } else {
        (c[1], c[0])
    };
    let (tr, br) = if c[3].1 > c[2].1 {
        (c[2], c[3])
    } else {
        (c[3], c[2])
    };
    [tl, tr, br, bl]
}

pub fn dist(a: Point, b: Point) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

/// DB "unclip": grow a rectangle by `d = area * ratio / perimeter` on every
/// side (what pyclipper's rounded offset reduces to once boxed again).
pub fn unclip(r: &Rect, ratio: f64) -> Option<Rect> {
    let [tl, tr, br, bl] = r.corners;
    let (w, h) = (dist(tl, tr), dist(tl, bl));
    if w == 0.0 || h == 0.0 {
        return None;
    }
    let d = w * h * ratio / (2.0 * (w + h));
    let u = ((tr.0 - tl.0) / w * d, (tr.1 - tl.1) / w * d);
    let v = ((bl.0 - tl.0) / h * d, (bl.1 - tl.1) / h * d);
    let corners = [
        (tl.0 - u.0 - v.0, tl.1 - u.1 - v.1),
        (tr.0 + u.0 - v.0, tr.1 + u.1 - v.1),
        (br.0 + u.0 + v.0, br.1 + u.1 + v.1),
        (bl.0 - u.0 + v.0, bl.1 - u.1 + v.1),
    ];
    Some(Rect {
        corners: order_corners(corners),
        short_side: (w + 2.0 * d).min(h + 2.0 * d),
    })
}

/// Point-in-convex-polygon (either winding, boundary inclusive).
pub fn inside(poly: &[Point; 4], p: Point) -> bool {
    let mut sign = 0.0;
    for i in 0..4 {
        let c = cross(poly[i], poly[(i + 1) % 4], p);
        if c != 0.0 {
            if sign != 0.0 && c.signum() != sign {
                return false;
            }
            sign = c.signum();
        }
    }
    true
}

/// Homography mapping `src[i] -> dst[i]` (4 point pairs), row-major 3x3.
pub fn homography(src: &[Point; 4], dst: &[Point; 4]) -> Option<[f64; 9]> {
    // Solve the 8x8 system A h = b for h = (h0..h7), h8 = 1.
    let mut a = [[0f64; 9]; 8];
    for i in 0..4 {
        let ((x, y), (u, v)) = (src[i], dst[i]);
        a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
        a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
    }
    for col in 0..8 {
        let pivot = (col..8).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        let pivot_row = a[col];
        for (r, row) in a.iter_mut().enumerate() {
            if r != col {
                let f = row[col] / pivot_row[col];
                for (v, p) in row.iter_mut().zip(pivot_row).skip(col) {
                    *v -= f * p;
                }
            }
        }
    }
    let h: Vec<f64> = (0..8).map(|i| a[i][8] / a[i][i]).collect();
    Some([h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0])
}

/// Rectify `quad` of `image` into a `w x h` image (`quad[0]` → top-left),
/// bilinear sampling with replicated borders (`cv2.warpPerspective` with
/// `BORDER_REPLICATE`).
pub fn warp_quad(image: &RgbImage, quad: &[Point; 4], w: u32, h: u32) -> RgbImage {
    let (wf, hf) = (w as f64, h as f64);
    let rect = [(0.0, 0.0), (wf, 0.0), (wf, hf), (0.0, hf)];
    let Some(m) = homography(&rect, quad) else {
        return RgbImage::new(w, h);
    };
    let (iw, ih) = (image.width() as i64, image.height() as i64);
    let px = |x: i64, y: i64| image.get_pixel(x.clamp(0, iw - 1) as u32, y.clamp(0, ih - 1) as u32);
    RgbImage::from_fn(w, h, |x, y| {
        let (x, y) = (x as f64, y as f64);
        let z = m[6] * x + m[7] * y + m[8];
        let sx = (m[0] * x + m[1] * y + m[2]) / z;
        let sy = (m[3] * x + m[4] * y + m[5]) / z;
        let (x0, y0) = (sx.floor(), sy.floor());
        let (fx, fy) = (sx - x0, sy - y0);
        let (x0, y0) = (x0 as i64, y0 as i64);
        let mut out = [0u8; 3];
        for (c, o) in out.iter_mut().enumerate() {
            let v = px(x0, y0)[c] as f64 * (1.0 - fx) * (1.0 - fy)
                + px(x0 + 1, y0)[c] as f64 * fx * (1.0 - fy)
                + px(x0, y0 + 1)[c] as f64 * (1.0 - fx) * fy
                + px(x0 + 1, y0 + 1)[c] as f64 * fx * fy;
            *o = v.round().clamp(0.0, 255.0) as u8;
        }
        Rgb(out)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Point, b: Point) -> bool {
        (a.0 - b.0).abs() < 1e-6 && (a.1 - b.1).abs() < 1e-6
    }

    #[test]
    fn components_are_8_connected_row_extremes() {
        #[rustfmt::skip]
        let m = [
            true,  true,  false, false,
            false, false, true,  false,
            false, false, false, false,
            false, false, false, true,
        ];
        let c = components(&m, 4, 4, 10);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0], vec![(0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (2.0, 1.0)]);
        assert_eq!(components(&m, 4, 4, 1).len(), 1);
    }

    #[test]
    fn min_area_rect_of_axis_aligned_and_rotated_boxes() {
        let r =
            min_area_rect(&[(1.0, 2.0), (9.0, 2.0), (9.0, 6.0), (1.0, 6.0), (5.0, 4.0)]).unwrap();
        assert_eq!(r.corners, [(1.0, 2.0), (9.0, 2.0), (9.0, 6.0), (1.0, 6.0)]);
        assert_eq!(r.short_side, 4.0);
        // 45° square of side sqrt(2).
        let r = min_area_rect(&[(1.0, 0.0), (2.0, 1.0), (1.0, 2.0), (0.0, 1.0)]).unwrap();
        assert!((r.short_side - 2f64.sqrt()).abs() < 1e-9);
        assert!(min_area_rect(&[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]).is_none());
    }

    #[test]
    fn unclip_grows_every_side_by_area_over_perimeter() {
        let r = Rect {
            corners: [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            short_side: 10.0,
        };
        // d = 100 * 1.5 / 40 = 3.75
        let u = unclip(&r, 1.5).unwrap();
        assert!(close(u.corners[0], (-3.75, -3.75)));
        assert!(close(u.corners[2], (13.75, 13.75)));
        assert_eq!(u.short_side, 17.5);
    }

    #[test]
    fn warp_identity_and_rotation() {
        let img = RgbImage::from_fn(4, 2, |x, y| Rgb([(x * 10 + y) as u8, 0, 0]));
        let same = warp_quad(
            &img,
            &[(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)],
            4,
            2,
        );
        assert_eq!(same, img);
        // Reading quad starting at the top-right corner, going down: the
        // crop is the image rotated 90° counter-clockwise.
        let rot = warp_quad(
            &img,
            &[(3.0, 0.0), (3.0, 2.0), (-1.0, 2.0), (-1.0, 0.0)],
            2,
            4,
        );
        assert_eq!(rot.get_pixel(0, 0)[0], 30);
        assert_eq!(rot.get_pixel(1, 0)[0], 31);
        assert_eq!(rot.get_pixel(0, 1)[0], 20);
        assert_eq!(rot.get_pixel(1, 3)[0], 1);
    }

    #[test]
    fn inside_is_boundary_inclusive() {
        let q = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)];
        assert!(inside(&q, (0.0, 2.0)) && inside(&q, (2.0, 2.0)));
        assert!(!inside(&q, (4.5, 2.0)));
    }
}
