# @zyzle/image-kmeans

Generate an array of colours from an image based on _k_-means clustering.

## Version 3.x changes and usage

Version 3.0 includes a lot of changes to optimization and performance.

Here's an example of usage in a React component

```tsx
import {
  type Color,
  ImageKmeans,
  type RunResult,
  default as init,
  InitMethod,
} from "@zyzle/image-kmeans";

function MyComponent() {
  const [wasmInstance, setWasmInstance] = useState<ImageKmeans | null>(null);

  useEffect(() => {
    init();
  }, []);

  // Drop handler 
  const handleDroppedFile = useCallback(
    async (ibm: ImageBitmap) => {
      const canvas = new OffscreenCanvas(ibm.width, ibm.height);
      const ctx = canvas.getContext(
        "2d"
      )! as unknown as CanvasRenderingContext2D;
      ctx.canvas.height = ibm.height;
      ctx.canvas.width = ibm.width;
      ctx.drawImage(ibm, 0, 0);
      const wasmInstance = new ImageKmeans(ctx, ibm.width, ibm.height);
      setWasmInstance(wasmInstance);
    },
    [setWasmInstance]
  );
}
```

### Run methods

### Fixed K number of clusters

Use a pre-determined number of clusters `k_number` for the calculations.

```ts
with_fixed_k_number(k_number: number, init_method: InitMethod, config: Config): Promise<RunResult>;
```

### Derived K number

The module will do multiple runs of the k-means algorithm and determine the
best fit for the number of selected clusters.

```ts
 with_derived_k_number(init_method: InitMethod, config: Config): Promise<RunResult>;
```

The `InitMethod` defines whether or not the initial clusters used in each run should be random or use the [K-means++](https://en.wikipedia.org/wiki/K-means++) algorithm.

The `Config` object used in both run types takes the following options:

| Option | Value |
|--------|-------|
| `quantize_fact` | clamp the colour space to given factor before calculating. Although this can significantly speed up performance it can result in the returned colors being close (usually indistinguishable at lower values) but not exactly those in the image |
| `top_num` | only consider the top *n* number of colours by frequency in calcultions |


> **Warning:** not providing either a `quantize_fact` or `top_num` configuration will result in every color in the image being used in calculations, this can take significantly longer to process on larger images with lots of colors like photographs


### Results object

Both of the above now return a RunResult object which looks like the following:

```ts
/**
 * Each \'run\' of the cluster calculation produces a result
 * containing the `k` size used, the vector of clusters found
 * and the within-cluster sum of squares (WCSS)
 */
export interface RunResult {
    /**
     * The number of `k` clusters used for this run
     */
    ks: number;
    /**
     * The cluster centroids found in this run
     */
    clusters: Color[];
    /**
     * The within-cluster sum of squares for this run
     */
    wcss: number;
}
```

## Building form source

> Due to changes in the getrandom crate you will either have to build with `RUSTFLAGS='--cfg getrandom_backend="wasm_js"'` or update your cargo configuration as described in [the getrandom docs](https://docs.rs/getrandom/0.3.3/getrandom/#opt-in-backends)

Use [`wasm-pack`](https://rustwasm.github.io/docs/wasm-pack/introduction.html) to build the Rust source into WebAssembly, this will output the JS/Wasm into a `pkg` folder using:

```bash
wasm-pack build --target web
# Or if you haven't made the updates to your cargo config
RUSTFLAGS='--cfg getrandom_backend="wasm_js"' wasm-pack build --target web
```

## License

Licensed under MIT license ([LICENSE](LICENSE) or http://opensource.org/licenses/MIT)
