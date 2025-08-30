mod utils;

use gloo_utils::format::JsValueSerdeExt;
use itertools::Itertools;
use rand::seq::IteratorRandom;
use serde::{Deserialize, Serialize};
use tsify::Tsify;
use utils::set_panic_hook;
use wasm_bindgen::prelude::*;
use web_sys::CanvasRenderingContext2d;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

/// Represents an RGB color
#[derive(Clone, Deserialize, Eq, Hash, PartialEq, Serialize, Tsify)]
pub struct Color {
    r: i32,
    g: i32,
    b: i32,
}

/// Each 'run' of the cluster calculation produces a result
/// containing the `k` size used, the vector of clusters found
/// and the within-cluster sum of squares (WCSS)
#[derive(Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct RunResult {
    ks: usize,
    clusters: Vec<Color>,
    wcss: f32,
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
    /// * `width` - The width of the rendered image
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

        let colors: Vec<Color> = pixels.into_iter().unique().collect();

        ImageKmeans {
            colors,
            initial_ks: vec![],
            results: vec![],
        }
    }

    /// Returns the ResultSet from the current run to JS as an array of `RunResult`s,
    /// in the case where no run has happend yet an empty array will be returned
    ///
    /// @deprecated
    pub fn get_result_set(&self) -> JsValue {
        JsValue::from_serde(&self.results).unwrap()
    }

    /// Do a run with a fixed number of `k` clusters and return the result set to JS
    /// as a single `RunResult`
    ///
    /// # Arguments
    ///
    /// * `k_number` - The number of `k` clusters to use for this run
    pub fn with_fixed_k_number(&mut self, k_number: usize) -> RunResult {
        self.use_random_ks(k_number);
        let result = self.do_run(k_number);

        self.results = vec![result];

        // JsValue::from_serde(&self.results[0]).unwrap()
        self.results[0].clone()
    }

    /// Performs multiple runs using `k` numbers between 1 and 20 and then uses
    /// analysis to determine the most appropriate number of `k` clusters to use
    /// for the provided image. Once determined the `RunResult` for this `k` number
    /// is returned
    pub fn with_derived_k_number(&mut self) -> RunResult {
        self.results = vec![];

        self.use_random_ks(10);

        for i in 1..=10 {
            self.results.push(self.do_run(i));
        }

        let wcss = self.results.iter().map(|r| r.wcss).collect::<Vec<f32>>();

        let (x1, y1) = (1.0, wcss[0]);
        let (x2, y2) = (11.0, wcss[wcss.len() - 1]);

        let mut distances: Vec<f32> = vec![];

        for (i, sum) in wcss.iter().enumerate() {
            let x0 = (i + 1) as f32;
            let y0 = *sum;
            let num = f32::abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1);
            let denum = f32::sqrt(f32::powi(y2 - y1, 2) + f32::powi(x2 - x1, 2));
            distances.push(num / denum);
        }

        let max_dist = distances.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let max_index = distances.iter().position(|&r| r == max_dist).unwrap();

        // JsValue::from_serde::<RunResult>(&self.results[max_index]).unwrap()
        self.results[max_index].clone()
    }
}

impl ImageKmeans {
    /// Take a random number of colors from the complete list of the given image
    /// and set these as `ImageKmeans.initial_ks`
    ///
    /// # Arguments
    /// * `a` - The number of random colors to pick for our initial k clusters
    fn use_random_ks(&mut self, a: usize) {
        let rng = &mut rand::rng();
        self.initial_ks = self.colors.clone().into_iter().choose_multiple(rng, a);
    }

    /// Perform a 'run' of the k-means clustering arlorithm taking a specified
    /// number of initial k colors from ImageKmeans.initial_ks
    ///
    /// # Arguments
    /// * `num_ks` - How many k clusters to run the algorithm for, these will be taken [0..num_ks]
    ///   from the ImageKmeans.initial_ks
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
            square_distance_sum = distance_sum;

            if distance_shift < 0_f32 || iterations == 10 {
                break;
            }

            iterations += 1;
            distance_shift = 0.0;
        }

        RunResult {
            ks: num_ks,
            clusters,
            wcss: square_distance_sum,
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

    /// Calculate the euclidean distance between two Color points in 3D space
    ///
    /// # Arguments
    /// * `p` - first color
    /// * `q` - second color
    fn calc_euclidean_dist(&self, p: &Color, q: &Color) -> f32 {
        f32::sqrt((i32::pow(p.r - q.r, 2) + i32::pow(p.g - q.g, 2) + i32::pow(p.b - q.b, 2)) as f32)
    }
}
