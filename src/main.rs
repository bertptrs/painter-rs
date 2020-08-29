use std::fs::File;
use std::path::PathBuf;

use structopt::StructOpt;

use painter_rs::Optimizer;

/// Simulate an evolutionary algorithm for generations.
///
/// For each generation, the best individual's score is printed and the individual is rendered to
/// the `output-dir` folder.
#[derive(StructOpt)]
struct Options {
    /// Input file. Must be 24-bit PNG without transparancy.
    #[structopt(parse(from_os_str))]
    input_file: PathBuf,

    /// Output directory for renders
    #[structopt(parse(from_os_str))]
    output_dir: PathBuf,

    /// Number of genrations to simulate for.
    #[structopt(short, long, default_value = "50")]
    generations: usize,

    /// Number of individuals in the population.
    #[structopt(short, long, default_value = "200")]
    population: usize,

    /// Ensure the best individuals survive the generational shift.
    ///
    /// If specified without argument, it will default to the population_size. Otherwise, specified
    /// number of individuals will survive into the next generation.
    #[structopt(short, long)]
    elitism: Option<Option<usize>>,
}

impl Options {
    pub fn elitism(&self) -> usize {
        self.elitism
            // Convert argument-less to population size
            .map(|v| v.unwrap_or(self.population))
            // Convert unspecified to 0
            .unwrap_or(0)
    }
}

fn main() {
    let options: Options = StructOpt::from_args();

    let mut input_file = File::open(&options.input_file).expect("Failed to read input file.");

    let mut optimizer = Optimizer::new(
        options.population,
        50,
        6,
        options.elitism(),
        &mut input_file,
    );

    for n in 0..options.generations {
        optimizer.advance();

        println!("Generation #{:04} best score: {}", n, optimizer.score(0));

        let target = options.output_dir.join(format!("{:04}.png", n));
        let mut target = File::create(target).expect("Failed to open output file");

        optimizer
            .draw_instance(0, &mut target)
            .expect("Failed to write output file");
    }
}
