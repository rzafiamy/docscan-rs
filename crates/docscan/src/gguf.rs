//! UVDoc checkpoint → one GGUF with the batch norms folded in, and loading.
//!
//! GGUF layout: `general.architecture = "docscan"`, `general.name`,
//! `docscan.model = "uvdoc"`, `docscan.input = [488, 712]` (w, h); tensors
//! `<conv>.weight` / `<conv>.bias` named as in [`crate::uvdoc::convs`] and
//! `out2d.prelu`. Weights F32 or F16, biases F32.
//!
//! The checkpoint (`best_model.pkl`, a `torch.save` zip) is read with
//! candle's pickle reader: tensors only, no code from the file runs.

use crate::uvdoc;
use anyhow::{Context, Result, bail};
use candle_core::quantized::{GgmlDType, QTensor, gguf_file};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

pub const ARCH: &str = "docscan";
const BN_EPS: f64 = 1e-5;

pub fn parse_dtype(name: &str) -> Result<GgmlDType> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "f32" => GgmlDType::F32,
        "f16" => GgmlDType::F16,
        other => bail!("unknown dtype '{other}' (f32, f16)"),
    })
}

/// Tensors of a UVDoc checkpoint (`model_state` dict, or a bare state dict,
/// or `.safetensors`), F32.
pub fn read_checkpoint(path: &Path) -> Result<HashMap<String, Tensor>> {
    let raw = if path.extension().is_some_and(|e| e == "safetensors") {
        candle_core::safetensors::load(path, &Device::Cpu)?
            .into_iter()
            .collect::<Vec<_>>()
    } else {
        let t = candle_core::pickle::read_all_with_key(path, Some("model_state"));
        match t {
            Ok(t) if !t.is_empty() => t,
            _ => candle_core::pickle::read_all(path)
                .with_context(|| format!("reading checkpoint {path:?}"))?,
        }
    };
    let mut out = HashMap::new();
    for (n, t) in raw {
        if n.ends_with("num_batches_tracked") {
            continue;
        }
        let n = n
            .strip_prefix("module.")
            .map(String::from)
            .unwrap_or(n.clone());
        out.insert(n, t.to_dtype(DType::F32)?);
    }
    if !out.contains_key("resnet_head.0.weight") {
        bail!("{path:?} holds no UVDoc weights (no resnet_head.0.weight)");
    }
    Ok(out)
}

/// Checkpoint names of a folded convolution: (conv prefix, batch-norm prefix).
fn source_of(name: &str) -> (String, Option<String>) {
    let s = |a: &str, b: Option<&str>| (a.to_string(), b.map(String::from));
    match name {
        "head.0" => s("resnet_head.0", Some("resnet_head.1")),
        "head.1" => s("resnet_head.3", Some("resnet_head.4")),
        "concat" => s("bridge_concat.0", Some("bridge_concat.1")),
        "out2d.0" => s("out_point_positions2D.0", Some("out_point_positions2D.1")),
        "out2d.1" => s("out_point_positions2D.3", None),
        _ => {
            if let Some(rest) = name.strip_prefix("bridge") {
                let (n, i) = rest.split_once('.').unwrap();
                let p = format!("bridge_{n}.{i}");
                return (format!("{p}.0"), Some(format!("{p}.1")));
            }
            // layerL.B.conv1 | conv2 | down
            let (lb, kind) = name.rsplit_once('.').unwrap();
            let p = format!("resnet_down.{lb}");
            match kind {
                "down" => (
                    format!("{p}.downsample.0"),
                    Some(format!("{p}.downsample.1")),
                ),
                k => {
                    let bn = if k == "conv1" { "bn1" } else { "bn2" };
                    (format!("{p}.{k}"), Some(format!("{p}.{bn}")))
                }
            }
        }
    }
}

/// Folds the checkpoint into the network's tensors (see module doc).
pub fn fold(ck: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    let get = |n: &str| -> Result<&Tensor> {
        ck.get(n)
            .with_context(|| format!("checkpoint has no '{n}'"))
    };
    let mut out = HashMap::new();
    for (name, spec) in uvdoc::convs() {
        let (conv, bn) = source_of(&name);
        // dilated convolutions sit in a Sequential: `conv1.0.weight`
        let wname = [format!("{conv}.weight"), format!("{conv}.0.weight")]
            .into_iter()
            .find(|n| ck.contains_key(n))
            .with_context(|| format!("checkpoint has no '{conv}.weight'"))?;
        let w = get(&wname)?.clone();
        let bname = wname.replace(".weight", ".bias");
        let b = match ck.get(&bname) {
            Some(b) => b.clone(),
            None => Tensor::zeros(spec.out_ch, DType::F32, &Device::Cpu)?,
        };
        let (w, b) = match bn {
            None => (w, b),
            Some(bn) => {
                let gamma = get(&format!("{bn}.weight"))?;
                let beta = get(&format!("{bn}.bias"))?;
                let mean = get(&format!("{bn}.running_mean"))?;
                let var = get(&format!("{bn}.running_var"))?;
                let scale = (gamma / (var + BN_EPS)?.sqrt()?)?;
                let w = w.broadcast_mul(&scale.reshape((spec.out_ch, 1, 1, 1))?)?;
                let b = ((b - mean)? * &scale)?.add(beta)?;
                (w, b)
            }
        };
        out.insert(format!("{name}.weight"), w);
        out.insert(format!("{name}.bias"), b);
    }
    out.insert(
        "out2d.prelu".into(),
        get("out_point_positions2D.2.weight")?.reshape(1)?,
    );
    // fails here, not at load, when the checkpoint is not UVDoc's shape
    uvdoc::UvDoc::load(&out, &Device::Cpu, DType::F32)
        .context("checkpoint does not match UVDoc")?;
    Ok(out)
}

/// Writes folded tensors to a GGUF; returns the tensor count per dtype.
pub fn write(
    tensors: &HashMap<String, Tensor>,
    out: &Path,
    dtype: GgmlDType,
    name: &str,
) -> Result<Vec<(String, usize)>> {
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();
    let mut q = Vec::new();
    let mut counts: HashMap<String, usize> = HashMap::new();
    for n in names {
        let t = &tensors[n];
        let dt = if t.rank() == 4 { dtype } else { GgmlDType::F32 };
        *counts.entry(format!("{dt:?}")).or_default() += 1;
        q.push((n.clone(), QTensor::quantize(t, dt)?));
    }
    let meta = [
        (
            "general.architecture",
            gguf_file::Value::String(ARCH.into()),
        ),
        ("general.name", gguf_file::Value::String(name.into())),
        ("docscan.model", gguf_file::Value::String("uvdoc".into())),
        (
            "docscan.input",
            gguf_file::Value::Array(vec![
                gguf_file::Value::U32(uvdoc::INPUT_W as u32),
                gguf_file::Value::U32(uvdoc::INPUT_H as u32),
            ]),
        ),
    ];
    let meta: Vec<(&str, &gguf_file::Value)> = meta.iter().map(|(k, v)| (*k, v)).collect();
    let refs: Vec<(&str, &QTensor)> = q.iter().map(|(n, t)| (n.as_str(), t)).collect();
    let mut file = std::io::BufWriter::new(std::fs::File::create(out)?);
    gguf_file::write(&mut file, &meta, &refs)?;
    let mut counts: Vec<_> = counts.into_iter().collect();
    counts.sort();
    Ok(counts)
}

/// Checkpoint → GGUF.
pub fn convert(
    ckpt: &Path,
    out: &Path,
    dtype: GgmlDType,
    name: &str,
) -> Result<Vec<(String, usize)>> {
    let folded = fold(&read_checkpoint(ckpt)?)?;
    write(&folded, out, dtype, name)
}

/// Loads a docscan GGUF: folded F32 tensors on the CPU, and the model name.
pub fn load(path: &Path) -> Result<(HashMap<String, Tensor>, String)> {
    let mut file = std::fs::File::open(path).with_context(|| format!("opening {path:?}"))?;
    let content =
        gguf_file::Content::read(&mut file).with_context(|| format!("reading {path:?}"))?;
    let meta = |k: &str| -> Option<String> {
        match content.metadata.get(k) {
            Some(gguf_file::Value::String(s)) => Some(s.clone()),
            _ => None,
        }
    };
    if meta("general.architecture").as_deref() != Some(ARCH) {
        bail!("{path:?} is not a {ARCH} GGUF");
    }
    if let Some(m) = meta("docscan.model")
        && m != "uvdoc"
    {
        bail!("{path:?}: unsupported docscan model '{m}' (this build: uvdoc)");
    }
    let name = meta("general.name").unwrap_or_default();
    let mut map = HashMap::new();
    let names: Vec<String> = content.tensor_infos.keys().cloned().collect();
    for n in names {
        let t = content.tensor(&mut file, &n, &Device::Cpu)?;
        map.insert(n, t.dequantize(&Device::Cpu)?);
    }
    Ok((map, name))
}
