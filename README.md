# @zyzle/image-kmeans

Generate an array of colours from an image based on _k_-means clustering.

## Version 2.x changes and usage

Version 2.0 comes with some big changes to the way this module is used. The process to run the code looks something like the following:

```js
import * as wasm from "@zyzle/image-kmeans";

const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");
// draw an image to the canvas

// Instantiate the class passing in your 2d rendering context and
// the image width and height
const wasmInstance = new ImageKmeans(ctx, ibm.width, ibm.height);
```

After instantiating the class you now have 2 choices

### Fixed K number of clusters

Use a pre-determined number of clusters for the calculations in this case 4:

```js
const result = wasmInstance.with_fixed_k_number(4);
```

### Derived K number

The module will do multiple runs of the k-means algorithm and determine the
best fit for the number of selected clusters.

**Note:** Because this will perform multiple complete runs of the algorithm it
may take significantly longer than a single run with a fixed number, although
should give better results, most of the time this will return with 40s even for
multi-megabyte images

```js
const result = wasmInstance.with_derived_k_number();
```

### Results object

Both of the above now return a RunResult object which looks like the following:

```
RunResult {
  ks: number;             // the number of k clusters used in this run
  clusters: Array<Color>; // An array containing color objects
                          // { r: number, g: number, b: number }
                          // representing the cluster centroids
  wcss: number            // the combined within-cluster sum of squares
                          // for these clusters
}
```

## Building form source

> Due to changes in the getrandom crate you will either have to build with `RUSTFLAGS='--cfg getrandom_backend="wasm_js"'` or update your cargo configuration as described in [the getrandom docs](https://docs.rs/getrandom/0.3.3/getrandom/#opt-in-backends)

Use [`wasm-pack`](https://rustwasm.github.io/docs/wasm-pack/introduction.html) to build the Rust source into WebAssembly, this will output the JS/Wasm into a `pkg` folder using:

```bash
wasm-pack build --target nodejs
# Or if you haven't made the updates to your cargo config
RUSTFLAGS='--cfg getrandom_backend="wasm_js"' wasm-pack build
```

## License

Licensed under MIT license ([LICENSE](LICENSE) or http://opensource.org/licenses/MIT)
