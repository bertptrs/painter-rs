use std::error::Error;
use std::io::{Read, Write};

use byte_slice_cast::AsSliceOf;
use cairo::{Context, Format, ImageSurface};
use rand_distr::{weighted::WeightedIndex, Binomial, Distribution, Normal, Standard};

pub struct Simulator {
    sides: usize,
    surface: ImageSurface,
}

impl Simulator {
    /// Type of image surface used for comparison and rendering.
    pub const IMAGE_FORMAT: Format = Format::Rgb24;

    pub fn new(sides: usize, width: usize, height: usize) -> Self {
        Self {
            sides,
            surface: ImageSurface::create(Self::IMAGE_FORMAT, width as i32, height as i32).unwrap(),
        }
    }

    pub fn simulate(&mut self, data: &[f64]) {
        // Mutability not strictly required but it avoids double-borrowing the surface.
        let width = self.surface.get_width() as f64;
        let height = self.surface.get_height() as f64;

        let ctx = Context::new(&self.surface);

        // Change transformation to avoid scaling every draw call.
        ctx.scale(width, height);

        // Clear the drawing board with white
        ctx.set_source_rgb(1.0, 1.0, 1.0);
        ctx.paint();

        let polygon_size = self.polygon_size();

        for polygon in data.chunks_exact(polygon_size) {
            let color = &polygon[..4];
            ctx.set_source_rgba(color[0], color[1], color[2], color[3]);

            ctx.new_path();

            for (i, (x, y)) in polygon[4..]
                .chunks_exact(2)
                // Safety: chunks_exact ensures that this slice is always large enough.
                .map(|p| unsafe { (*p.get_unchecked(0), *p.get_unchecked(1)) })
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

        self.surface.flush();
    }

    pub fn polygon_size(&self) -> usize {
        // two x per point plus 4 for the color.
        2 * self.sides + 4
    }

    pub fn write_buffer(&self, dest: &mut impl Write) -> Result<(), Box<dyn Error>> {
        self.surface.write_to_png(dest)?;
        Ok(())
    }

    /// Compute the sum of the square pixel difference between images.
    ///
    /// This image surface must be of the same type as is used to render the image and should be
    /// of equal dimensions.
    ///
    /// # Panics
    /// If the supplied `ImageSurface` is not compatible with the surface used, this method will
    /// panic.
    pub fn compare(&mut self, to: &mut ImageSurface) -> f64 {
        // First: sanity checks on the provided image.
        assert_eq!(
            to.get_format(),
            Self::IMAGE_FORMAT,
            "Image format should match"
        );

        assert_eq!(
            to.get_width(),
            self.surface.get_width(),
            "Image dimensions should match."
        );

        assert_eq!(
            to.get_height(),
            self.surface.get_height(),
            "Image dimensions should match."
        );

        let stride1 = self.surface.get_stride();
        let stride2 = to.get_stride();
        let width = to.get_width() as usize;

        let data1 = self.surface.get_data().unwrap();
        let data2 = to.get_data().unwrap();

        let mut diff = 0.0;

        // Future idea: use cairo's OPERATOR_DIFFERENCE to compute the difference and sum the result.
        for (row1, row2) in data1
            .chunks_exact(stride1 as usize)
            .zip(data2.chunks_exact(stride2 as usize))
        {
            // Can be safely unwrapped as cairo ensures alignment.
            let row1 = row1.as_slice_of::<u32>().unwrap();
            let row2 = row2.as_slice_of::<u32>().unwrap();

            for pixels in row1.iter().copied().zip(row2.iter().copied()).take(width) {
                let (mut c1, mut c2) = pixels;

                // Rgb24 uses the lower 24 bits for the channels and the upper 8 are undefined.
                for _ in 0..3 {
                    let v1 = c1 & 0xff;
                    c1 >>= 8;
                    let v2 = c2 & 0xff;
                    c2 >>= 8;
                    let delta = v1.max(v2) - v1.min(v2);
                    diff += (delta * delta) as f64;
                }
            }
        }

        diff
    }
}

pub struct Optimizer {
    target: ImageSurface,
    sim: Simulator,
    population: Vec<Vec<f64>>,
    mutation_chance: Binomial,
    mutation_amount: Normal<f64>,
}

impl Optimizer {
    pub fn new(
        pop_size: usize,
        polygon_count: usize,
        sides: usize,
        target: &mut impl Read,
    ) -> Self {
        // TODO: experiment with other initial distributions and RNG.
        let dist = Standard;
        let rng = rand::thread_rng();

        // TODO: better error handling.
        let target = ImageSurface::create_from_png(target).expect("Failed to read PNG target.");
        let sim = Simulator::new(
            sides,
            target.get_width() as usize,
            target.get_height() as usize,
        );

        // Initialize random initial population
        let population = (0..pop_size)
            .map(|_| {
                dist.sample_iter(rng)
                    .take(sim.polygon_size() * polygon_count)
                    .collect()
            })
            .collect();

        Self {
            target,
            sim,
            population,
            // TODO: tweak
            mutation_chance: Binomial::new(1, 0.2).unwrap(),
            mutation_amount: Normal::new(0.0, 0.01).unwrap(),
        }
    }

    fn evaluate(&mut self) -> Vec<f64> {
        // Manual borrows to ensure the checker approves of the lambda.
        let sim = &mut self.sim;
        let target = &mut self.target;

        self.population
            .iter()
            .map(|instance| {
                sim.simulate(instance);
                sim.compare(target)
            })
            .collect()
    }

    pub fn advance(&mut self) {
        let scores = self.evaluate();

        let parent_distribution = WeightedIndex::new(scores.iter().map(|&s| 1. / s)).unwrap();
        let pop_size = self.population.len();

        let new_population = (0..pop_size)
            .map(|_| self.generate_offspring(&parent_distribution))
            .collect();

        self.population = new_population;
    }

    fn generate_offspring(&self, parents: impl Distribution<usize>) -> Vec<f64> {
        let mut rng = rand::thread_rng();

        let parent1 = &self.population[parents.sample(&mut rng)];
        let parent2 = &self.population[parents.sample(&mut rng)];

        let size = parent1.len();

        // Should be correct, but a serious bug if it isn't.
        debug_assert_eq!(size, parent2.len());

        // TODO: experiment with different crossover types.
        // For now just split the genome in the middle.
        let crossover_point = size / 2;

        let mut offspring = Vec::with_capacity(size);

        offspring.extend_from_slice(&parent1[..crossover_point]);
        offspring.extend_from_slice(&parent2[crossover_point..]);

        // Check if indeed we generated a valid offspring.
        debug_assert_eq!(size, offspring.len());

        self.mutate(&mut offspring);

        offspring
    }

    fn mutate(&self, instance: &mut [f64]) {
        let mut rng = rand::thread_rng();

        instance
            .iter_mut()
            .filter(move |_| self.mutation_chance.sample(&mut rng) == 1)
            .for_each(move |n| *n = (*n + self.mutation_amount.sample(&mut rng)).max(0.).min(1.));
    }
}

#[cfg(test)]
mod tests {
    use std::env::temp_dir;
    use std::fs::File;

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

    #[test]
    fn compute_difference() {
        let mut sim = Simulator::new(6, 10, 10);
        sim.simulate(&[]);

        let mut reference = ImageSurface::create(Simulator::IMAGE_FORMAT, 10, 10).unwrap();

        // Create a blank canvas
        let ctx = Context::new(&reference);
        ctx.set_source_rgb(1.0, 1.0, 1.0);
        ctx.paint();
        drop(ctx);

        // They should be equal now
        let difference = sim.compare(&mut reference);

        assert_eq!(0.0, difference);

        // Now paint it blue
        let ctx = Context::new(&reference);
        ctx.set_source_rgb(0.0, 0.0, 1.0);
        ctx.paint();
        drop(ctx);

        // Difference should be (255^2 + 255^2) * 10 * 10
        let difference = sim.compare(&mut reference);
        assert_eq!(13005000., difference);
    }

    #[test]
    fn run_optimizer() {
        let mut optimizer = Optimizer::new(
            10,
            10,
            6,
            &mut include_bytes!("../samples/rustacean-small.png").as_ref(),
        );
        optimizer.advance();
    }
}
