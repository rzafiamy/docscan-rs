//! One-page PDF holding one image: JPEG (DCTDecode) for colour and gray,
//! Flate for black and white (exact pixels). No dependency beyond flate2.

use crate::image::Image;
use anyhow::Result;
use flate2::Compression;
use flate2::write::ZlibEncoder;
use std::io::Write;
use std::str::FromStr;

/// Page size: the image at its DPI, or the image fitted (centred, aspect
/// kept) on a paper size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Paper {
    #[default]
    Auto,
    A4,
    Letter,
}

impl FromStr for Paper {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> anyhow::Result<Self> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "" => Paper::Auto,
            "a4" => Paper::A4,
            "letter" => Paper::Letter,
            o => anyhow::bail!("unknown paper '{o}' (auto, a4, letter)"),
        })
    }
}

impl Paper {
    /// Points (1/72 in), portrait.
    fn points(self) -> Option<(f64, f64)> {
        match self {
            Paper::Auto => None,
            Paper::A4 => Some((595.276, 841.89)),
            Paper::Letter => Some((612.0, 792.0)),
        }
    }
}

/// Builds the PDF. `jpeg`: the image's JPEG bytes when it is to be embedded
/// as JPEG (colour / gray); `None` embeds the exact pixels, Flate-compressed.
pub fn write(img: &Image, jpeg: Option<&[u8]>, dpi: f64, paper: Paper) -> Result<Vec<u8>> {
    let (iw, ih) = (
        img.width as f64 * 72.0 / dpi,
        img.height as f64 * 72.0 / dpi,
    );
    let (pw, ph, dw, dh) = match paper.points() {
        None => (iw, ih, iw, ih),
        Some((a, b)) => {
            // page orientation follows the image
            let (pw, ph) = if iw > ih { (b, a) } else { (a, b) };
            let s = (pw / iw).min(ph / ih);
            (pw, ph, iw * s, ih * s)
        }
    };
    let (ox, oy) = ((pw - dw) / 2.0, (ph - dh) / 2.0);
    let color = if img.channels == 3 {
        "/DeviceRGB"
    } else {
        "/DeviceGray"
    };
    let (filter, data) = match jpeg {
        Some(j) => ("/DCTDecode", j.to_vec()),
        None => {
            let mut z = ZlibEncoder::new(Vec::new(), Compression::default());
            z.write_all(&img.bytes())?;
            ("/FlateDecode", z.finish()?)
        }
    };
    let content = format!("q {dw:.3} 0 0 {dh:.3} {ox:.3} {oy:.3} cm /Im0 Do Q\n");

    let mut out: Vec<u8> = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n".to_vec();
    let mut offsets = Vec::new();
    let mut obj = |out: &mut Vec<u8>, body: &[u8]| {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n", offsets.len()).as_bytes());
        out.extend_from_slice(body);
        out.extend_from_slice(b"\nendobj\n");
    };
    obj(&mut out, b"<< /Type /Catalog /Pages 2 0 R >>");
    obj(&mut out, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(
        &mut out,
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw:.3} {ph:.3}] \
             /Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>"
        )
        .as_bytes(),
    );
    let mut image = format!(
        "<< /Type /XObject /Subtype /Image /Width {} /Height {} /ColorSpace {color} \
         /BitsPerComponent 8 /Filter {filter} /Length {} >>\nstream\n",
        img.width,
        img.height,
        data.len()
    )
    .into_bytes();
    image.extend_from_slice(&data);
    image.extend_from_slice(b"\nendstream");
    obj(&mut out, &image);
    let mut stream = format!("<< /Length {} >>\nstream\n", content.len()).into_bytes();
    stream.extend_from_slice(content.as_bytes());
    stream.extend_from_slice(b"endstream");
    obj(&mut out, &stream);
    obj(&mut out, b"<< /Producer (docscan) >>");

    let xref = out.len();
    out.extend_from_slice(
        format!("xref\n0 {}\n0000000000 65535 f \n", offsets.len() + 1).as_bytes(),
    );
    for o in &offsets {
        out.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R /Info {} 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            offsets.len() + 1,
            offsets.len()
        )
        .as_bytes(),
    );
    Ok(out)
}
