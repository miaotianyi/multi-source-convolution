# multi-source-convolution
A PyTorch 2d convolution block that simultaneously takes image, vector, and scalar data.

## Motivation
What if an image comes with useful scalar metadata?
For example, in image signal processing, each raw image comes
with a camera ISO scalar, which can be useful.
How can we make a CNN simultaneously take image and scalar data?

`MSConv2d` can be used out of the box and is faster than
the naive `tile_concat.py`.

## Tile, Concatenate, and Convolve
In `tile_concat.py`, I implement a naive version of tile-concat-conv paradigm.
This allows us to combine scalars, vectors, and images into the same input tensor.

### Tile
- `[N, C1, H, W]` image: remain the same
- `[N, C2, H]` vector: unsqueeze to `[N, C2, H, 1]`, tile to `[N, C2, H, W]`
- `[N, C3, W]` vector: unsqueeze to `[N, C3, 1, W]`, tile to `[N, C3, H, W]`
- `[N, C4]` scalar: unsqueeze to `[N, C4, 1, 1]`, tile to `[N, C4, H, W]`

Tiling is done with `torch.expand` to avoid copying the underlying tensors in memory.

### Concatenate
The 4 resulting tensors from above are concatenated into
a single `[N, C1+C2+C3+C4, H, W]` tensor.

### Convolve
The resulting `[N, C1+C2+C3+C4, H, W]` tensor is passed to
a classic `nn.Conv2d` convolution block.

### Applications
This tile-concat-conv operation can be performed at the beginning,
in the middle, and at the end of a network.
The `[N, C4]` input scalars can come from an MLP
or a text embedding network, not necessarily some raw input scalars.

### Should we flatten vectors?
It's also possible to flatten `[N, C2, H]` to `[N, C2 * H]`
and `[N, C3, W]` to `[N, C3 * W]` and treat them as scalar inputs.
If the vectors indeed encode row-wise or column-wise metadata,
this flattening typically worsens performance.

## Multi-source Convolution
This is my main contribution based on the original formula for convolution.
It also implements tile-concat-conv paradigm while being faster
and more memory-efficient.

Let `C` be the number of output channels, 
`MSConv2d` handles each input tensors as follows:
- `[N, C1, H, W]` image: 2d convolution along H and W, produces `[N, C, H, W]`
- `[N, C2, H]` vector: 1d convolution along H, produces `[N, C, H]`
- `[N, C3, W]` vector: 1d convolution along H, produces `[N, C, W]`
- `[N, C4]` scalar: fully connected layer (`nn.Linear`), produces `[N, C]`

The kernel sizes are adjusted accordingly for the 1d convolutions.
Also, only one of the above layers need bias.

These output tensors then unsqueeze to
`[N, C, H, W], [N, C, H, 1], [N, C, 1, W], [N, C, 1, 1]` respectively.
They are then added with broadcasting,
producing `[N, C, H, W]` output.

The core design comes from the observation that
after naive tiling, the input tensor to the `nn.Conv2d` block
will have lots of constant channels (all elements in the channels are the same).
In that case, `kernel_size * kernel_size` parameters convolve with a constant
receptive field, so only 1 parameter is necessary to represent the same computation.
This saves lots of computation time and `nn.Conv2d` weights.

## Equivalence and Efficiency
For any set of `nn.Conv2d` weights in a vanilla tile-concat-conv architecture,
there's a `MSConv2d` block with weights that
result in the same inference-time computational output.
However, this doesn't guarantee that `MSConv2d` will learn these equivalent weights
when it's in the same position as 
`nn.Conv2d` with the same hyperparameters following `tile_and_concatenate`.
Let me know if there are significant differences.
I haven't found any.

I like to compare this to quadratic forms $x^TAx$.
We always assume that $A$ is a symmetric matrix,
because in the context of quadratic forms,
even if $A$ is not symmetric, we can easily produce $(A+A^T)/2$,
which gives the same output for all $x$.

`MSConv2d` is maximally efficient because the scalars and vectors
don't need to repeatedly go through convolution kernels.
