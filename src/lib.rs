mod utils;

use gloo_utils::format::JsValueSerdeExt;
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

/// Represents an RGB color
#[derive(Clone, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct Color {
    r: i32,
    g: i32,
    b: i32,
}

/// Each 'run' of the cluster calculation produces a result
/// containing the `k` size used, the vector of clusters found
/// and the within-cluster sum of squares (WCSS)
#[derive(Clone, Serialize)]
struct RunResult {
    ks: usize,
    clusters: Vec<Color>,
    sum: f32,
}

/// Represents the instance of the module containing the current images
/// pixel Colors and the last set of RunResults
#[wasm_bindgen]
#[derive(Serialize)]
pub struct ImageKmeans {
    colors: Vec<Color>,
    initial_ks: Vec<Color>,
    results: Vec<RunResult>,
}

#[wasm_bindgen]
impl ImageKmeans {
    /// Creates an instance of the ImageKmeans. This is also decorated as
    /// the JS constructor so will be run directly as a call to
    ///
    /// ```js
    /// const kmeans = new ImageKmeans(ctx, height, width);
    /// ```
    ///
    /// # Arguments
    ///
    /// * `ctx` - The canvas 2d rendering context containing the image
    /// * `width` - The width of the rendred image
    /// * `height` - the height of the rendered image
    #[wasm_bindgen(constructor)]
    pub fn new(ctx: &CanvasRenderingContext2d, width: u32, height: u32) -> ImageKmeans {
        set_panic_hook();
        let image_data = ctx
            .get_image_data(0.0, 0.0, width as f64, height as f64)
            .unwrap();
        let color_data = image_data.data();
        let mut pixels: Vec<Color> = vec![];

        for i in (0..color_data.len()).step_by(4) {
            pixels.push(Color {
                r: color_data[i] as i32,
                g: color_data[i + 1] as i32,
                b: color_data[i + 2] as i32,
            });
        }

        let colors: Vec<Color> = pixels.into_iter().duplicates().collect();

        ImageKmeans {
            colors,
            initial_ks: vec![],
            results: vec![],
        }
    }

    /// Returns the ResultSet from the current run to JS as an array of `RunResult`s,
    /// in the case where no run has happend yet an empty array will be returned
    pub fn get_result_set(&self) -> JsValue {
        JsValue::from_serde(&self.results).unwrap()
    }

    /// Do a run with a fixed number of `k` clusters and return the result set to JS
    /// as an array containing a single `RunResult`
    ///
    /// # Arguments
    ///
    /// * `k_number` - The number of `k` clusters to use for this run
    pub fn with_fixed_k_number(&mut self, k_number: usize) -> JsValue {
        self.use_random_ks(k_number);
        let result = self.do_run(k_number);

        self.results = vec![result];

        JsValue::from(
            self.results[0]
                .clusters
                .iter()
                .map(|c| format!("#{:02x}{:02x}{:02x}", c.r, c.g, c.b))
                .map(JsValue::from)
                .collect::<js_sys::Array>(),
        )
    }

    /// Performs multiple runs using `k` numbers between 1 and 10 and then uses
    /// analysis to determine the most appropriate number of `k` clusters to use
    /// for the provided image. Once determined the `RunResult` for this `k` number
    /// is returned
    pub fn with_derrived_k_number(&mut self) -> JsValue {
        self.results = vec![];

        self.use_random_ks(10);

        for i in 1..=10 {
            self.results.push(self.do_run(i));
        }

        let points: Vec<(f32, f32)> = self.results.iter().map(|p| (p.ks as f32, p.sum)).collect();
        let b = self.calc_euclidean_dist_2d(&points[0], &points[points.len() - 1]) as f64;
        let mut hs = vec![];
        // for each point in points:
        for (i, ps) in points.iter().enumerate() {
            if i == 0 || i == (points.len() - 1) {
                hs.push((0, 0.0));
            } else {
                let a: f64 = self.calc_euclidean_dist_2d(&points[0], &ps) as f64;
                let c: f64 = self.calc_euclidean_dist_2d(&ps, &points[points.len() - 1]) as f64;
                let p: f64 = (a + b + c) as f64;
                let s = p / 2.0_f64;
                let t = f64::sqrt(s * ((s - a) * (s - b) * (s - c)));
                let h = 2_f64 * (t / b);
                hs.push((i, h));
                console::log_1(&format!("{:?}", (s, t, h)).into());
            }
        }

        console::log_1(&format!("{:?}", hs).into());

        let maxh = hs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b.1));
        let hindex = hs.iter().position(|&r| r.1 == maxh).unwrap();
        console::log_1(&format!("hindex: {}", hindex).into());
        //   a = calc_e_d(points[0], point)
        //   c = calc_e_d(point, points[last])
        //   p = a + b + c
        //   s = p / 2
        //   A = sqrt(s(s-a)(s-b)(s-c))
        //   h = 2 * (A /b)

        // largest H is elbow and therefore best?

        console::log_1(&JsValue::from_serde(&self.results[hs[hindex].0]).unwrap());

        JsValue::from_serde(&self.results[hs[hindex].0]).unwrap()
    }
}

impl ImageKmeans {
    fn use_random_ks(&mut self, a: usize) -> () {
        let rng = &mut rand::thread_rng();
        self.initial_ks = self.colors.clone().into_iter().choose_multiple(rng, a);
    }

    fn do_run(&self, num_ks: usize) -> RunResult {
        let mut iterations = 0;
        #[allow(unused_assignments)]
        let mut square_distance_sum = 0.0;
        let mut distance_shift = 0.0;

        let mut clusters = self.initial_ks[..num_ks].to_vec();

        loop {
            let (new_clusters, distance_sum) = self.calc_new_clusters(&clusters);

            for i in 0..new_clusters.len() {
                distance_shift += self.calc_euclidean_dist(&new_clusters[i], &clusters[i])
            }

            distance_shift /= new_clusters.len() as f32;
            clusters = new_clusters;
            square_distance_sum = distance_sum; // / num_ks as f32;

            if distance_shift < 1_f32 || iterations == 10 {
                break;
            }

            iterations += 1;
            distance_shift = 0.0;
        }

        RunResult {
            ks: num_ks,
            clusters,
            sum: square_distance_sum,
        }
    }

    fn calc_new_clusters(&self, k_clusters: &Vec<Color>) -> (Vec<Color>, f32) {
        let mut new_clusters = vec![vec![]; k_clusters.len()];

        for color in &self.colors {
            let distances = k_clusters
                .iter()
                .map(|k| self.calc_euclidean_dist(k, color))
                .collect::<Vec<f32>>();

            let min_distance = distances.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let selected_k = distances.iter().position(|&r| r == min_distance).unwrap();
            new_clusters[selected_k].push(color);
        }

        let colors: Vec<Color> = new_clusters
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
            .collect();

        let distance_sum = colors
            .iter()
            .zip(new_clusters)
            .map(|(a, b)| {
                let mut sum_total = 0.0;
                for c in b {
                    sum_total += self.calc_euclidean_dist(a, c).powi(2);
                }
                sum_total
            })
            .sum();

        (colors, distance_sum)
    }

    fn calc_euclidean_dist(&self, p: &Color, q: &Color) -> f32 {
        f32::sqrt((i32::pow(p.r - q.r, 2) + i32::pow(p.g - q.g, 2) + i32::pow(p.b - q.b, 2)) as f32)
    }

    fn calc_euclidean_dist_2d(&self, p: &(f32, f32), q: &(f32, f32)) -> f32 {
        f32::sqrt(f32::powi(p.0 - q.0, 2) + f32::powi(p.1 - q.1, 2))
    }
}
