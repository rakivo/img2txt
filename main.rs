use image::GenericImageView;
use tiny_http::{Response, Request, Header, Server, Method};
use ocrs::{ImageSource, OcrEngineParams, OcrEngine, TextItem};

const HTML_BYTES: &[u8] = include_bytes!("index.html");
const DETECTION_BYTES: &[u8] = include_bytes!("text-detection.rten");
const RECOGNITION_BYTES: &[u8] = include_bytes!("text-recognition.rten");

const PORT: &str = "6969";
const ADDR: &str = "localhost";

#[inline]
fn serve_bytes(request: Request, bytes: &[u8], content_type: &str) -> anyhow::Result::<()> {
    let content_type_header = Header::from_bytes("Content-Type", content_type).unwrap();
    request.respond(Response::from_data(bytes).with_header(content_type_header))?;
    Ok(())
}

#[inline]
fn img_src_from_bytes<R>(r: &mut R) -> anyhow::Result::<ImageSource>
where
    R: std::io::Read + ?Sized
{
    let mut bytes = Vec::new();
    r.read_to_end(&mut bytes)?;
    let img = image::load_from_memory(&bytes)?;
    let dims = img.dimensions();
    let img_src = ImageSource::from_bytes(img.into_rgb8().into_raw().leak(), dims)?;
    Ok(img_src)
}

fn text_from_img_src(engine: &OcrEngine, img_src: ImageSource) -> anyhow::Result::<String> {
    let ocr_input = engine.prepare_input(img_src)?;
    let word_rects = engine.detect_words(&ocr_input)?;
    let line_rects = engine.find_text_lines(&ocr_input, &word_rects);
    let text = engine.recognize_text(&ocr_input, &line_rects)?
        .into_iter()
        .flatten()
        .filter_map(|l| {
            let chars = l.chars();
            if !l.chars().is_empty() {
                Some(chars.into_iter().map(|c| c.char).collect::<String>())
            } else {
                None
            }
        }).collect();
    Ok(text)
}

fn main() -> anyhow::Result::<()> {
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

    for mut rq in server.incoming_requests() {
        match (rq.method(), rq.url()) {
            (&Method::Post, "/img2txt") => {
                let img_src = img_src_from_bytes(rq.as_reader())?;
                let string = text_from_img_src(&engine, img_src)?;
                serve_bytes(rq, string.as_bytes(), "text/plain")
            }
            _ => serve_bytes(rq, HTML_BYTES, "text/html; charset=UTF-8")
        }?;
    }

    Ok(())
}
