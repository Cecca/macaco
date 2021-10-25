use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion,
    PlotConfiguration,
};
use kcmkc_base::types::*;
use kcmkc_sequential::kcenter::kcenter;

fn bench_kcenter(c: &mut Criterion) {
    let data: Vec<Higgs> = kcmkc_base::dataset::Dataset::new(
        "/Users/matteoceccarello/Work/kcmkc/.datasets/higgs/higgs-v1.msgpack.gz",
    )
    .to_vec(None)
    .unwrap();

    let k = 8;

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("kcenter");
    group.plot_config(plot_config);
    for frac in [0.1, 0.2, 0.4, 0.8, 1.0].iter() {
        let n: usize = (data.len() as f64 * frac) as usize;
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| kcenter(&data[0..n], k));
        });
    }
}

criterion_group!(benches, bench_kcenter);
criterion_main!(benches);
