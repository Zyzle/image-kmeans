mod utils;

use itertools::Itertools;
use rand::seq::IteratorRandom;
use serde_derive::{Deserialize, Serialize};
use utils::set_panic_hook;
use wasm_bindgen::prelude::*;
use web_sys::{console, CanvasRenderingContext2d};

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[derive(Clone, Deserialize, Eq, Hash, PartialEq, Serialize)]
struct Color {
    r: i32,
    g: i32,
    b: i32,
}

#[wasm_bindgen]
pub fn find_colors(ctx: &CanvasRenderingContext2d, width: u32, height: u32) -> JsValue {
    set_panic_hook();

    console::time_with_label("grabbing image data");
    let image_data = ctx
        .get_image_data(0.0, 0.0, width as f64, height as f64)
        .unwrap();

    let color_data = image_data.data();
    console::time_end_with_label("grabbing image data");

    let mut pixels: Vec<Color> = vec![];

    console::time_with_label("dataset iterator");
    for i in (0..color_data.len()).step_by(4) {
        pixels.push(Color {
            r: color_data[i] as i32,
            g: color_data[i + 1] as i32,
            b: color_data[i + 2] as i32,
        });
    }
    console::time_end_with_label("dataset iterator");

    console::time_with_label("unique");
    let colors: Vec<Color> = pixels.into_iter().unique().collect();
    console::time_end_with_label("unique");

    let rng = &mut rand::thread_rng();

    console::time_with_label("pick 8");
    let mut clusters = colors.clone().into_iter().choose_multiple(rng, 8);
    console::time_end_with_label("pick 8");

    let mut iterations = 0;
    let mut distance_shift = 0.0;

    loop {
        console::time_with_label("Calc new clusters");
        let new_clusters = calc_new_clusters(&clusters, &colors);
        console::time_end_with_label("Calc new clusters");

        for i in 0..new_clusters.len() {
            distance_shift += calc_euclidean_dist(&new_clusters[i], &clusters[i])
        }

        distance_shift /= new_clusters.len() as f32;

        clusters = new_clusters;

        if distance_shift <= 5_f32 || iterations >= 10 {
            break;
        }

        iterations += 1;
        distance_shift = 0_f32;
    }

    JsValue::from(
        clusters
            .iter()
            .map(|c| format!("#{:02x}{:02x}{:02x}", c.r, c.g, c.b))
            .map(JsValue::from)
            .collect::<js_sys::Array>(),
    )
}

fn calc_new_clusters(k_clusters: &Vec<Color>, color_data: &Vec<Color>) -> Vec<Color> {
    let mut new_clusters = vec![vec![]; k_clusters.len()];

    for color in color_data {
        let distances = k_clusters
            .iter()
            .map(|k| calc_euclidean_dist(k, color))
            .collect::<Vec<f32>>();

        let min_distance = distances.iter().fold(f32::INFINITY, |a, &b| a.min(b));

        let selected_k = distances.iter().position(|&r| r == min_distance).unwrap();

        new_clusters[selected_k].push(color);
    }

    new_clusters
        .iter()
        .map(|c_list| {
            let mut r = 0;
            let mut b = 0;
            let mut g = 0;

            c_list.iter().for_each(|color| {
                r += color.r;
                b += color.b;
                g += color.g;
            });

            Color {
                r: (r / c_list.len() as i32),
                g: (g / c_list.len() as i32),
                b: (b / c_list.len() as i32),
            }
        })
        .collect()
}

fn calc_euclidean_dist(p: &Color, q: &Color) -> f32 {
    f32::sqrt((i32::pow(p.r - q.r, 2) + i32::pow(p.g - q.g, 2) + i32::pow(p.b - q.b, 2)) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclid_zero() {
        let result = calc_euclidean_dist(&Color { r: 0, g: 0, b: 0 }, &Color { r: 0, g: 0, b: 0 });

        assert_eq!(result, 0_f32);
    }

    #[test]
    fn test_euclid() {
        let result = calc_euclidean_dist(&Color { r: 1, g: 1, b: 0 }, &Color { r: 2, g: 1, b: 2 });

        assert_eq!(result, 2.236068_f32);
    }
}
