use std::error::Error;
use std::io::Write;

use cairo::{Context, Format, ImageSurface};

pub struct Simulator {
    sides: usize,
    surface: ImageSurface,
}

impl Simulator {
    pub fn new(sides: usize, width: usize, height: usize) -> Self {
        Self {
            sides,
            surface: ImageSurface::create(Format::ARgb32, width as i32, height as i32).unwrap(),
        }
    }

    pub fn simulate(&mut self, data: &[f64]) {
        // Mutability not strictly required but it avoids double-borrowing the surface.
        let width = self.surface.get_width() as f64;
        let height = self.surface.get_height() as f64;

        let ctx = Context::new(&self.surface);

        // Clear the drawing board with white
        ctx.new_path();
        ctx.set_source_rgb(1.0, 1.0, 1.0);
        ctx.rectangle(0.0, 0.0, width, height);
        ctx.fill();
        ctx.close_path();

        let polygon_size = self.polygon_size();

        for polygon in data.chunks_exact(polygon_size) {
            let color = &polygon[..4];
            ctx.set_source_rgba(color[0], color[1], color[2], color[3]);

            ctx.new_path();

            for (i, [x, y]) in polygon[4..]
                .chunks_exact(2)
                .map(|s| [s[0] * width, s[1] * height])
                .enumerate()
            {
                if i == 0 {
                    ctx.move_to(x, y);
                } else {
                    ctx.line_to(x, y);
                }
            }

            ctx.fill();

            ctx.close_path();
        }
    }

    pub fn polygon_size(&self) -> usize {
        // two x per point plus 4 for the color.
        2 * self.sides + 4
    }

    pub fn write_buffer(&self, dest: &mut impl Write) -> Result<(), Box<dyn Error>> {
        self.surface.write_to_png(dest)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::env::temp_dir;
    use std::fs::File;

    use rand::distributions::{Distribution, Standard};

    use super::*;

    /// Write a random test file out for a random instance.
    ///
    /// This serves as a test for the entire simulation pipeline.
    #[test]
    fn write_random_file() {
        let mut sim = Simulator::new(6, 200, 200);

        let instance: Vec<f64> = Standard
            .sample_iter(rand::thread_rng())
            .take(50 * sim.polygon_size())
            .collect();

        sim.simulate(&instance);

        let filename = temp_dir().join("test.png");

        sim.write_buffer(&mut File::create(&filename).unwrap())
            .unwrap();

        // clean up after ourselves
        std::fs::remove_file(&filename).unwrap();
    }
}
