# @zyzle/image-kmeans

Generate an array of colours from an image based on _k_-means clustering. 

## Building form source

Use [`wasm-pack`](https://rustwasm.github.io/docs/wasm-pack/introduction.html) to build the Rust source into WebAssembly, this will output the JS/Wasm into a `pkg` folder using:

```bash
wasm-pack build
```

## Usage

The module can be imported into JS with:

```js
import * as wasm from '@zyzle/image-kmeans';
```

Remember if including this directly in an HTML file your module will have to be loaded as a single async import:

```js
import("./index.js")
  .catch(e => console.error("Error importing `index.js`:", e));
```

Currently, the wasm has a single exported function `find_colours` a minimal usage example would be as follows:

```js
import * as wasm from '@zyzle/image-kmeans';

const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");

// draw an image to the canvas

const result = wasm.find_colors(ctx, imageWidth, imageHeight);
```

Once processing has finished `result` will contain a `String[]` of the colours in hex notation.

## License

Licensed under MIT license ([LICENSE](LICENSE) or http://opensource.org/licenses/MIT)
