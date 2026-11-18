//! UVDoc (Verhoeven, Magne, Sorkine-Hornung, SIGGRAPH Asia 2023): a dilated
//! ResNet predicting a 45×31 grid of source positions for the unwarped
//! document. Inference only, 2-D head only (the 3-D head is a training
//! signal), batch norms folded into the convolutions by `docscan convert`.
//!
//! Spec: `model.py` of github.com/tanguymagne/UVDoc (num_filter 32, kernel 5).

use crate::warp::Grid;
use anyhow::{Context, Result, bail};
use candle_core::{D, DType, Device, Tensor};
use std::collections::HashMap;

/// Network input, width × height (UVDoc `IMG_SIZE`).
pub const INPUT_W: usize = 488;
pub const INPUT_H: usize = 712;
/// Output grid, columns × rows (UVDoc `GRID_SIZE` is [45, 31] = rows, columns).
pub const GRID_W: usize = 31;
pub const GRID_H: usize = 45;

/// One convolution of the folded network.
#[derive(Clone, Copy, Debug)]
pub struct ConvSpec {
    pub out_ch: usize,
    pub in_ch: usize,
    pub kernel: usize,
    pub stride: usize,
    pub dilation: usize,
    /// Zero padding; `None`: reflect padding of kernel / 2 (output head).
    pub padding: Option<usize>,
}

fn conv(out_ch: usize, in_ch: usize, kernel: usize, stride: usize, dilation: usize) -> ConvSpec {
    ConvSpec {
        out_ch,
        in_ch,
        kernel,
        stride,
        dilation,
        padding: Some(dilation * (kernel / 2)),
    }
}

/// Residual stages: (blocks, channels, first stride).
pub const STAGES: [(usize, usize, usize); 3] = [(3, 32, 1), (4, 64, 2), (6, 128, 2)];
/// Bridge dilations (each list one sequential branch).
pub const BRIDGES: [&[usize]; 6] = [&[1], &[2], &[5], &[8, 3, 2], &[12, 7, 4], &[18, 12, 6]];

/// Every convolution of the folded network, by GGUF name (`<name>.weight`,
/// `<name>.bias`), in forward order; plus `out2d.prelu` (one value).
pub fn convs() -> Vec<(String, ConvSpec)> {
    let mut v = vec![
        ("head.0".to_string(), conv(32, 3, 5, 2, 1)),
        ("head.1".to_string(), conv(32, 32, 5, 2, 1)),
    ];
    let mut in_ch = 32;
    for (s, &(blocks, ch, stride)) in STAGES.iter().enumerate() {
        for b in 0..blocks {
            let p = format!("layer{}.{b}", s + 1);
            if b == 0 {
                // is_top: plain convolutions, the first one strided
                v.push((format!("{p}.conv1"), conv(ch, in_ch, 5, stride, 1)));
                v.push((format!("{p}.conv2"), conv(ch, ch, 5, 1, 1)));
                if stride != 1 || in_ch != ch {
                    v.push((format!("{p}.down"), conv(ch, in_ch, 5, stride, 1)));
                }
            } else {
                v.push((format!("{p}.conv1"), conv(ch, ch, 5, 1, 3)));
                v.push((format!("{p}.conv2"), conv(ch, ch, 5, 1, 3)));
            }
        }
        in_ch = ch;
    }
    for (n, ds) in BRIDGES.iter().enumerate() {
        for (i, &d) in ds.iter().enumerate() {
            v.push((format!("bridge{}.{i}", n + 1), conv(128, 128, 3, 1, d)));
        }
    }
    v.push(("concat".into(), conv(128, 128 * 6, 1, 1, 1)));
    let reflect = |o, i| ConvSpec {
        padding: None,
        ..conv(o, i, 5, 1, 1)
    };
    v.push(("out2d.0".into(), reflect(32, 128)));
    v.push(("out2d.1".into(), reflect(2, 32)));
    v
}

struct Conv {
    w: Tensor,
    b: Tensor,
    spec: ConvSpec,
}

impl Conv {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let s = self.spec;
        let (x, pad) = match s.padding {
            Some(p) => (x.clone(), p),
            None => (pad_reflect(x, s.kernel / 2)?, 0),
        };
        let y = x.conv2d(&self.w, pad, s.stride, s.dilation, 1)?;
        Ok(y.broadcast_add(&self.b)?)
    }
}

/// Reflect padding of the last two dimensions (PyTorch `padding_mode="reflect"`).
fn pad_reflect(x: &Tensor, p: usize) -> Result<Tensor> {
    let mut x = x.clone();
    for dim in [2, 3] {
        let n = x.dim(dim)?;
        if p >= n {
            bail!("reflect padding {p} on a dimension of {n}");
        }
        let idx: Vec<u32> = (1..=p)
            .rev()
            .chain(0..n)
            .chain((n - 1 - p..n - 1).rev())
            .map(|i| i as u32)
            .collect();
        let idx = Tensor::new(idx.as_slice(), x.device())?;
        x = x.index_select(&idx, dim)?;
    }
    Ok(x)
}

pub struct UvDoc {
    convs: HashMap<String, Conv>,
    prelu: Tensor,
    pub device: Device,
    pub dtype: DType,
}

impl UvDoc {
    /// Builds the network from folded F32 tensors (any device: moved here).
    pub fn load(tensors: &HashMap<String, Tensor>, device: &Device, dtype: DType) -> Result<Self> {
        let get = |n: &str| -> Result<Tensor> {
            tensors
                .get(n)
                .with_context(|| format!("missing tensor '{n}'"))?
                .to_device(device)?
                .to_dtype(dtype)
                .map_err(Into::into)
        };
        let mut convs = HashMap::new();
        for (name, spec) in self::convs() {
            let w = get(&format!("{name}.weight"))?;
            let want = [spec.out_ch, spec.in_ch, spec.kernel, spec.kernel];
            if w.dims() != want {
                bail!("{name}.weight: shape {:?}, expected {want:?}", w.dims());
            }
            let b = get(&format!("{name}.bias"))?;
            if b.dims() != [spec.out_ch] {
                bail!(
                    "{name}.bias: shape {:?}, expected [{}]",
                    b.dims(),
                    spec.out_ch
                );
            }
            let b = b.reshape((1, spec.out_ch, 1, 1))?;
            convs.insert(name, Conv { w, b, spec });
        }
        let prelu = get("out2d.prelu")?.reshape(())?;
        Ok(Self {
            convs,
            prelu,
            device: device.clone(),
            dtype,
        })
    }

    fn c(&self, name: &str, x: &Tensor) -> Result<Tensor> {
        self.convs[name].forward(x)
    }

    /// `x`: (B, 3, H, W) RGB in 0..1 (H, W: [`INPUT_H`], [`INPUT_W`] for the
    /// trained model). Returns (B, 2, GRID_H, GRID_W): x, y in -1..1.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.to_dtype(self.dtype)?;
        let x = self.c("head.0", &x)?.relu()?;
        let mut x = self.c("head.1", &x)?.relu()?;
        for (s, &(blocks, _, _)) in STAGES.iter().enumerate() {
            for b in 0..blocks {
                let p = format!("layer{}.{b}", s + 1);
                let down = format!("{p}.down");
                let res = if self.convs.contains_key(&down) {
                    self.c(&down, &x)?
                } else {
                    x.clone()
                };
                let y = self.c(&format!("{p}.conv1"), &x)?.relu()?;
                let y = self.c(&format!("{p}.conv2"), &y)?;
                x = (y + res)?.relu()?;
            }
        }
        let mut branches = Vec::with_capacity(BRIDGES.len());
        for (n, ds) in BRIDGES.iter().enumerate() {
            let mut y = x.clone();
            for i in 0..ds.len() {
                y = self.c(&format!("bridge{}.{i}", n + 1), &y)?.relu()?;
            }
            branches.push(y);
        }
        let x = Tensor::cat(&branches, 1)?;
        let x = self.c("concat", &x)?.relu()?;
        let x = self.c("out2d.0", &x)?;
        // PReLU, one shared slope
        let neg = x.minimum(0.0)?.broadcast_mul(&self.prelu)?;
        let x = (x.relu()? + neg)?;
        Ok(self.c("out2d.1", &x)?.to_dtype(DType::F32)?)
    }

    /// Runs one image (already resized to the input size, HWC RGB) and
    /// returns its grid.
    pub fn predict(&self, img: &crate::image::Image) -> Result<Grid> {
        if img.channels != 3 {
            bail!("UVDoc needs an RGB image");
        }
        let x = Tensor::from_slice(&img.data, (1, img.height, img.width, 3), &Device::Cpu)?
            .permute((0, 3, 1, 2))?
            .contiguous()?
            .to_device(&self.device)?;
        let g = self.forward(&x)?.squeeze(0)?; // (2, gh, gw)
        let (gh, gw) = (g.dim(1)?, g.dim(2)?);
        let g = g.permute((1, 2, 0))?.flatten_all()?.to_vec1::<f32>()?;
        let points = g.chunks_exact(2).map(|p| [p[0], p[1]]).collect();
        Ok(Grid { gw, gh, points })
    }
}

/// Max |a - b| / max |b| over two tensors of the same shape.
pub fn rel_err(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?;
    let d = (a - &b)?.abs()?.max(D::Minus1)?.to_scalar::<f32>()?;
    let m = b.abs()?.max(D::Minus1)?.to_scalar::<f32>()?;
    Ok(d / m.max(1e-12))
}
