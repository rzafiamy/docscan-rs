//! Float images (HWC, values in 0..=1), decoding with the EXIF orientation
//! applied, PNG / JPEG encoding.

use anyhow::{Context, Result, bail};
use image::{DynamicImage, ImageDecoder, ImageReader};
use std::io::Cursor;

/// A float image, rows top to bottom, channels interleaved (1 = gray, 3 = RGB).
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub data: Vec<f32>,
}

impl Image {
    pub fn new(width: usize, height: usize, channels: usize) -> Self {
        Self {
            width,
            height,
            channels,
            data: vec![0.0; width * height * channels],
        }
    }

    pub fn from_vec(width: usize, height: usize, channels: usize, data: Vec<f32>) -> Result<Self> {
        if data.len() != width * height * channels {
            bail!(
                "image data: {} values for {width}x{height}x{channels}",
                data.len()
            );
        }
        Ok(Self {
            width,
            height,
            channels,
            data,
        })
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize, c: usize) -> f32 {
        self.data[(y * self.width + x) * self.channels + c]
    }

    /// Rec. 601 luma of an RGB image (a gray image is returned as is).
    pub fn to_gray(&self) -> Image {
        if self.channels == 1 {
            return self.clone();
        }
        let data = self
            .data
            .chunks_exact(self.channels)
            .map(|p| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2])
            .collect();
        Image {
            width: self.width,
            height: self.height,
            channels: 1,
            data,
        }
    }

    fn to_u8(&self) -> Vec<u8> {
        self.data
            .iter()
            .map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
            .collect()
    }

    fn to_dynamic(&self) -> Result<DynamicImage> {
        let (w, h) = (self.width as u32, self.height as u32);
        Ok(match self.channels {
            1 => DynamicImage::ImageLuma8(
                image::GrayImage::from_raw(w, h, self.to_u8()).context("gray buffer")?,
            ),
            3 => DynamicImage::ImageRgb8(
                image::RgbImage::from_raw(w, h, self.to_u8()).context("rgb buffer")?,
            ),
            c => bail!("cannot encode a {c}-channel image"),
        })
    }

    pub fn encode_png(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.to_dynamic()?
            .write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)?;
        Ok(out)
    }

    /// Baseline JPEG; `dpi` goes in the JFIF header.
    pub fn encode_jpeg(&self, quality: u8, dpi: u16) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        let mut enc =
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality.clamp(1, 100));
        enc.set_pixel_density(image::codecs::jpeg::PixelDensity::dpi(dpi));
        enc.encode_image(&self.to_dynamic()?)?;
        Ok(out)
    }

    /// 8-bit samples, as PDF and the encoders want them.
    pub fn bytes(&self) -> Vec<u8> {
        self.to_u8()
    }
}

/// Decodes JPEG, PNG, WebP, TIFF or BMP; applies the EXIF orientation
/// (phone photos are often stored sideways); drops alpha. Returns RGB.
pub fn decode(bytes: &[u8]) -> Result<Image> {
    let reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .context("reading image")?;
    if reader.format().is_none() {
        bail!("unsupported image format (JPEG, PNG, WebP, TIFF, BMP)");
    }
    let mut decoder = reader.into_decoder().context("decoding image")?;
    let orientation = decoder.orientation().ok();
    let mut img = DynamicImage::from_decoder(decoder).context("decoding image")?;
    if let Some(o) = orientation {
        img.apply_orientation(o);
    }
    let rgb = img.to_rgb32f();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    Image::from_vec(w, h, 3, rgb.into_raw())
}
