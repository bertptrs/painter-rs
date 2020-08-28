use std::fs::File;
use std::path::PathBuf;

use structopt::StructOpt;

use painter_rs::Optimizer;

#[derive(StructOpt)]
struct Options {
    /// Input file
    #[structopt(parse(from_os_str))]
    input_file: PathBuf,

    /// Output directory for renders
    #[structopt(parse(from_os_str))]
    output_dir: PathBuf,

    #[structopt(short, long, default_value = "50")]
    generations: usize,
}

fn main() {
    let options: Options = StructOpt::from_args();

    let mut input_file = File::open(&options.input_file).expect("Failed to read input file.");

    let mut optimizer = Optimizer::new(200, 50, 6, &mut input_file);

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
