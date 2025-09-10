#![warn(clippy::all, clippy::pedantic)]
#![no_std]
extern crate alloc;

mod utils;

use alloc::{
    boxed::Box, collections::btree_map::BTreeMap, format, string::ToString, vec, vec::Vec,
};
use rand::seq::IteratorRandom;
use serde::{Deserialize, Serialize};
use tsify::Tsify;
use utils::set_panic_hook;
use wasm_bindgen::prelude::*;
use web_sys::CanvasRenderingContext2d;

/// Represents an RGB color
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Tsify)]
pub struct Color {
    /// The red component of the color [0-255]
    r: i32,
    /// The green component of the color [0-255]
    g: i32,
    /// The blue component of the color [0-255]
    b: i32,
}

/// Each 'run' of the cluster calculation produces a result
/// containing the `k` size used, the vector of clusters found
/// and the within-cluster sum of squares (WCSS)
#[derive(Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct RunResult {
    /// The number of `k` clusters used for this run
    ks: usize,
    /// The cluster centroids found in this run
    clusters: Vec<Color>,
    /// The within-cluster sum of squares for this run
    wcss: f32,
}

/// Configuration options for a run of the k-means algorithm
#[derive(Clone, Serialize, Debug, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct Config {
    /// An optional factor to quantize the colors by before running
    #[tsify(optional)]
    quantize_fact: Option<i32>,
    /// Only consider this number of the most frequent colors for clustering
    #[tsify(optional)]
    top_num: Option<usize>,
}

/// Represents the instance of the module containing the current images
/// pixel Colors and the last set of `RunResults`
#[wasm_bindgen]
#[derive(Serialize)]
pub struct ImageKmeans {
    colors: Vec<Color>,
    working_colors: Option<Vec<Color>>,
    working_colors_counts: Option<BTreeMap<Color, usize>>,
    initial_ks: Vec<Color>,
    results: Vec<RunResult>,
}

/// The method to use to pick the initial `k` clusters
#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
#[wasm_bindgen]
pub enum InitMethod {
    /// Randomly select initial `k` clusters
    Random,
    /// Use the K-means++ algorithm to select initial `k` clusters
    KmeansPlusPlus,
}

const ALLOWED_ERROR_DISTANCE_CMP: f32 = 0.1;

#[wasm_bindgen]
impl ImageKmeans {
    /// Creates an instance of the `ImageKmeans`. This is also decorated as
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
    ///
    /// # Panics
    /// Panics if the image data cannot be retrieved from the provided context
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(ctx: &CanvasRenderingContext2d, width: u32, height: u32) -> Self {
        set_panic_hook();
        let image_data = ctx
            .get_image_data(0.0, 0.0, f64::from(width), f64::from(height))
            .unwrap();
        let color_data = image_data.data();
        let mut colors: Vec<Color> = vec![];

        for i in (0..color_data.len()).step_by(4) {
            colors.push(Color {
                r: i32::from(color_data[i]),
                g: i32::from(color_data[i + 1]),
                b: i32::from(color_data[i + 2]),
            });
        }

        ImageKmeans {
            colors,
            working_colors: None,
            working_colors_counts: None,
            initial_ks: vec![],
            results: vec![],
        }
    }

    /// Do a run with a fixed number of `k` clusters and return the result set to JS
    /// as a single `RunResult`
    ///
    /// # Arguments
    /// * `k_number` - The number of `k` clusters to use for this run
    /// * `init_method` - The method to use to pick the initial `k` clusters
    /// * `config` - Configuration options for the run
    ///   * `quantize_fact` - An optional factor to quantize the colors by before running
    ///   * `top_num` - Only consider this number of the most frequent colors for clustering
    #[allow(clippy::unused_async)]
    pub async fn with_fixed_k_number(
        &mut self,
        k_number: usize,
        init_method: InitMethod,
        config: Config,
    ) -> RunResult {
        self.set_working_colors(config.quantize_fact, config.top_num);

        match init_method {
            InitMethod::Random => self.use_random_ks(k_number),
            InitMethod::KmeansPlusPlus => self.use_kmeans_plus_plus(k_number),
        }
        let result = self.do_run(k_number);

        self.results = vec![result];

        self.results[0].clone()
    }

    /// Performs multiple runs using `k` numbers between 1 and 10 and then uses
    /// analysis to determine the most appropriate number of `k` clusters to use
    /// for the provided image. Once determined the `RunResult` for this `k` number
    /// is returned
    ///
    /// # Arguments
    /// * `init_method` - The method to use to pick the initial `k` clusters
    /// * `config` - Configuration options for the run
    ///   * `quantize_fact` - An optional factor to quantize the colors by before running
    ///   * `top_num` - Only consider this number of the most frequent colors for clustering
    /// # Returns
    /// The `RunResult` for the determined optimal `k` number
    /// # Panics
    /// Panics if no results are found (should not happen)
    /// or if the maximum distance calculation fails (should not happen)
    /// when determining the optimal `k` number
    #[allow(clippy::unused_async, clippy::cast_precision_loss)]
    pub async fn with_derived_k_number(
        &mut self,
        init_method: InitMethod,
        config: Config,
    ) -> RunResult {
        self.set_working_colors(config.quantize_fact, config.top_num);

        match init_method {
            InitMethod::Random => self.use_random_ks(10),
            InitMethod::KmeansPlusPlus => self.use_kmeans_plus_plus(10),
        }

        self.results = vec![];

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

        let max_index = distances
            .iter()
            .position(|&r| (r - max_dist).abs() < ALLOWED_ERROR_DISTANCE_CMP)
            .unwrap();

        self.results[max_index].clone()
    }

    /// Set the working colors to be used for clustering, this takes the complete color list
    /// and applies the optional quantization and top number filtering
    /// # Arguments
    /// * `quantize_fact` - An optional factor to quantize the colors by before running
    /// * `top_num` - Only consider this number of the most frequent colors for clustering
    fn set_working_colors(&mut self, quantize_fact: Option<i32>, top_num: Option<usize>) {
        let mut color_counts: BTreeMap<Color, usize> = BTreeMap::new();

        for color in &self.colors {
            *color_counts
                .entry(quantize(color, quantize_fact.unwrap_or(1)))
                .or_insert(0) += 1;
        }

        self.working_colors_counts = Some(color_counts.clone());

        let mut color_vec: Vec<_> = color_counts.iter().collect();
        color_vec.sort_by(|a, b| b.1.cmp(a.1));
        let colors: Vec<Color> = color_vec
            .iter()
            .take(top_num.unwrap_or(color_counts.len()))
            .map(|(c, _)| (*c).clone())
            .collect();

        self.working_colors = Some(colors);
    }

    /// Take a random number of colors from the complete list of the given image
    /// and set these as `ImageKmeans.initial_ks`
    ///
    /// # Arguments
    /// * `a` - The number of random colors to pick for our initial k clusters
    fn use_random_ks(&mut self, a: usize) {
        let rng = &mut rand::rng();
        self.initial_ks = self
            .working_colors
            .as_ref()
            .unwrap()
            .clone()
            .into_iter()
            .choose_multiple(rng, a);
    }

    /// Use the Kmeans++ algorithm to pick initial k clusters
    ///
    /// # Arguments
    /// * `a` - The number of initial k clusters to pick
    fn use_kmeans_plus_plus(&mut self, a: usize) {
        let mut rng = &mut rand::rng();
        let first_k = self
            .working_colors
            .as_ref()
            .unwrap()
            .clone()
            .into_iter()
            .choose(&mut rng)
            .unwrap();
        let mut k_clusters = vec![first_k.clone()];
        let mut distances: Vec<f32> =
            vec![f32::INFINITY; self.working_colors.as_ref().unwrap().len()];

        let mut colors = self.working_colors.as_ref().unwrap().clone();
        colors.retain(|c| *c != first_k);

        while k_clusters.len() < a {
            // Update distances: for each color, keep the minimum distance to any cluster center
            for (i, color) in colors.iter().enumerate() {
                let mut min_dist = f32::INFINITY;
                for center in &k_clusters {
                    let dist = calc_euclidean_dist(center, color);
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }
                distances[i] = min_dist;
            }

            // Choose next center probabilistically proportional to squared distance
            let dist_sum: f32 = distances.iter().sum();
            if dist_sum == 0.0 {
                // All distances are zero, pick random
                if let Some(next_k) = colors.iter().choose(&mut rng) {
                    let next_k_cloned = next_k.clone();
                    k_clusters.push(next_k_cloned.clone());
                    colors.retain(|c| *c != next_k_cloned);
                    distances = vec![f32::INFINITY; colors.len()];
                } else {
                    break;
                }
            } else {
                let mut probs: Vec<f32> = distances.iter().map(|d| d / dist_sum).collect();
                for i in 1..probs.len() {
                    probs[i] += probs[i - 1];
                }
                let r: f32 = rand::random();
                let mut next_k_index = 0;
                for (i, p) in probs.iter().enumerate() {
                    if r < *p {
                        next_k_index = i;
                        break;
                    }
                }
                let next_k = colors[next_k_index].clone();
                k_clusters.push(next_k.clone());
                let next_k_cloned = next_k.clone();
                colors.retain(|c| *c != next_k_cloned);
                distances = vec![f32::INFINITY; colors.len()];
            }
        }
        self.initial_ks = k_clusters;
    }

    /// Perform a 'run' of the k-means clustering algorithm taking a specified
    /// number of initial k colors from `ImageKmeans.initial_ks`
    ///
    /// # Arguments
    /// * `num_ks` - How many k clusters to run the algorithm for, these will be taken [`0..num_ks`]
    ///   from the `ImageKmeans.initial_ks`
    /// # Returns
    /// The `RunResult` for this run
    #[allow(clippy::cast_precision_loss)]
    fn do_run(&self, num_ks: usize) -> RunResult {
        let mut iterations = 0;
        #[allow(unused_assignments)]
        let mut square_distance_sum = 0.0;
        let mut distance_shift = 0.0;

        let max_ks = num_ks.min(self.initial_ks.len());

        let mut clusters = self.initial_ks[..max_ks].to_vec();

        loop {
            let (new_clusters, distance_sum) = self.calc_new_clusters(&clusters);

            for i in 0..new_clusters.len() {
                distance_shift += calc_euclidean_dist(&new_clusters[i], &clusters[i]);
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

    /// Given a set of k clusters, calculate the new clusters by assigning each color
    /// to the nearest cluster and then recalculating the cluster centroids
    /// # Arguments
    /// * `k_clusters` - The current k clusters to use as centroids
    /// # Returns
    /// A tuple containing the new clusters and the within-cluster sum of squares
    /// for these clusters
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_possible_wrap
    )]
    fn calc_new_clusters(&self, k_clusters: &[Color]) -> (Vec<Color>, f32) {
        let mut new_clusters = vec![vec![]; k_clusters.len()];

        for color in self.working_colors.as_ref().unwrap() {
            let distances = k_clusters
                .iter()
                .map(|k| calc_euclidean_dist(k, color))
                .collect::<Vec<f32>>();

            let min_distance = distances.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            // let selected_k = distances.iter().position(|&r| r == min_distance).unwrap();
            let selected_k = distances
                .iter()
                .position(|&r| (r - min_distance).abs() < ALLOWED_ERROR_DISTANCE_CMP)
                .unwrap();
            new_clusters[selected_k].push(color);
        }

        let colors: Vec<Color> = new_clusters
            .iter()
            .map(|c_list| {
                if c_list.is_empty() {
                    // If cluster is empty, fallback to a default (could random or zero)
                    return Color { r: 0, g: 0, b: 0 };
                }
                // Compute mean color
                let mut r_sum = 0;
                let mut b_sum = 0;
                let mut g_sum = 0;
                let mut total_count = 0;

                for color in c_list {
                    let count = self
                        .working_colors_counts
                        .as_ref()
                        .unwrap()
                        .get(color)
                        .copied()
                        .unwrap_or(1);
                    r_sum += color.r * count as i32;
                    b_sum += color.b * count as i32;
                    g_sum += color.g * count as i32;
                    total_count += count;
                }

                let mean = Color {
                    r: r_sum / total_count as i32,
                    g: g_sum / total_count as i32,
                    b: b_sum / total_count as i32,
                };

                // Find the color in the cluster closest to the mean
                // taking the color_counts into account
                // Break ties by weighted distance
                c_list
                    .iter()
                    .min_by(|a, b| {
                        let wa = self
                            .working_colors_counts
                            .as_ref()
                            .unwrap()
                            .get(a)
                            .copied()
                            .unwrap_or(1) as f32;
                        let wb = self
                            .working_colors_counts
                            .as_ref()
                            .unwrap()
                            .get(b)
                            .copied()
                            .unwrap_or(1) as f32;
                        let da = calc_euclidean_dist(a, &mean) / wa;
                        let db = calc_euclidean_dist(b, &mean) / wb;
                        da.partial_cmp(&db).unwrap()
                    })
                    .copied()
                    .unwrap()
                    .clone()
            })
            .collect();

        let distance_sum = colors
            .iter()
            .zip(new_clusters)
            .map(|(a, b)| {
                let mut sum_total = 0.0;
                for c in b {
                    sum_total += calc_euclidean_dist(a, c).powi(2);
                }
                sum_total
            })
            .sum();

        (colors, distance_sum)
    }
}

/// Calculate the euclidean distance between two Color points in 3D space
///
/// # Arguments
/// * `p` - first color
/// * `q` - second color
#[allow(clippy::cast_precision_loss)]
fn calc_euclidean_dist(p: &Color, q: &Color) -> f32 {
    f32::sqrt(
        ((p.r - q.r) * (p.r - q.r) + (p.g - q.g) * (p.g - q.g) + (p.b - q.b) * (p.b - q.b)) as f32,
    )
}

/// Quantize a `Color` by reducing its precision by the given factor
/// # Arguments
/// * `color` - The `Color` to quantize
/// * `factor` - The factor to reduce the color precision by
fn quantize(color: &Color, factor: i32) -> Color {
    Color {
        r: (color.r / factor) * factor,
        g: (color.g / factor) * factor,
        b: (color.b / factor) * factor,
    }
}
