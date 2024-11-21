use anyhow::Result;
use if_chain::if_chain;
use stb_image::stb_image;
use rayon::iter::{ParallelBridge, ParallelIterator};
use ocrs::{ImageSource, OcrEngineParams, OcrEngine};
use tiny_http::{Response, Request, Header, Server, Method};

const HTML_BYTES: &[u8] = include_bytes!("index.html");
const DETECTION_BYTES: &[u8] = include_bytes!("text-detection.rten");
const RECOGNITION_BYTES: &[u8] = include_bytes!("text-recognition.rten");

const PORT: &str = "6969";
const ADDR: &str = "localhost";

type Dims = (u32, u32);

#[inline]
fn serve_bytes(request: Request, bytes: &[u8], content_type: &str) -> Result::<()> {
    let content_type_header = Header::from_bytes("Content-Type", content_type).unwrap();
    request.respond(Response::from_data(bytes).with_header(content_type_header))?;
    Ok(())
}

fn stbi_read_image(bytes: &[u8]) -> Result::<(&[u8], Dims)> {
    // assuming rgb
    const CHANNELS: i32 = 3;

    let mut w = 0;
    let mut h = 0;
    let data_ptr = unsafe {
        stb_image::stbi_load_from_memory(
            bytes.as_ptr(),
            bytes.len() as _,
            &mut w,
            &mut h,
            std::ptr::null_mut(),
            CHANNELS
        )
    };

    if data_ptr.is_null() {
        return Err(anyhow::anyhow!("failed to load image from memory"))
    }

    let data_len = (w * h * CHANNELS) as usize;
    let data_slice = unsafe { std::slice::from_raw_parts(data_ptr, data_len) };

    Ok((data_slice, (w as _, h as _)))
}

#[inline]
fn img_src_from_bytes<R>(r: &mut R, cap: usize) -> anyhow::Result::<ImageSource>
where
    R: std::io::Read + ?Sized
{
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(cap)?;
    r.read_to_end(&mut bytes)?;
    let (data, dims) = stbi_read_image(bytes.leak())?;
    let img_src = ImageSource::from_bytes(data, dims)?;
    Ok(img_src)
}

fn main() -> Result::<()> {
    let addr = format!("{ADDR}:{PORT}");
    let server = Server::http(&addr).unwrap();

    println!("serving at: http://{addr}");

    let detection_model = rten::Model::load_static_slice(DETECTION_BYTES)?;
    let recognition_model = rten::Model::load_static_slice(RECOGNITION_BYTES)?;

    let engine = OcrEngine::new(OcrEngineParams {
        detection_model: Some(detection_model),
        recognition_model: Some(recognition_model),
        ..Default::default()
    })?;

    server.incoming_requests().par_bridge().for_each(|mut rq| unsafe {
        match (rq.method(), rq.url()) {
            (&Method::Post, "/img2txt") => if_chain! {
                if let Some(body_len) = rq.body_length();
                if let Ok(img_src) = img_src_from_bytes(rq.as_reader(), body_len);
                if let Ok(ocr_input) = engine.prepare_input(img_src);
                if let Ok(string) = engine.get_text(&ocr_input);
                then {
                    serve_bytes(rq, string.as_bytes(), "text/plain")
                } else {
                    Ok(())
                }
            },
            _ => serve_bytes(rq, HTML_BYTES, "text/html; charset=UTF-8")
        }.unwrap_unchecked();
    });

    Ok(())
}
