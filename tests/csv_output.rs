use std::fs;
use std::sync::Arc;
use std::path::Path;
use arrow::array::*;
use crossbow::{DataFrame, Series, read_csv, write_csv};

static OUTPUT_DIR: &str = "tests/output";

fn ensure_output_dir() {
    let _ = fs::create_dir_all(OUTPUT_DIR);
}

fn load_test_data() -> DataFrame {
    read_csv("data_1M.csv").unwrap()
}

fn make_int_mask(series: &Series, pred: fn(i64) -> bool) -> Series {
    if let Some(arr) = series.as_primitive::<Int32Array>() {
        let mut b = BooleanArray::builder(arr.len());
        for i in 0..arr.len() {
            b.append_value(arr.is_valid(i) && pred(arr.value(i) as i64));
        }
        return Series::new("mask", Arc::new(b.finish()));
    }
    let arr = series.as_primitive::<Int64Array>().unwrap();
    let mut b = BooleanArray::builder(arr.len());
    for i in 0..arr.len() {
        b.append_value(arr.is_valid(i) && pred(arr.value(i)));
    }
    Series::new("mask", Arc::new(b.finish()))
}

fn make_float_mask(series: &Series, pred: fn(f64) -> bool) -> Series {
    let arr = series.as_primitive::<Float64Array>().unwrap();
    let mut b = BooleanArray::builder(arr.len());
    for i in 0..arr.len() {
        b.append_value(arr.is_valid(i) && pred(arr.value(i)));
    }
    Series::new("mask", Arc::new(b.finish()))
}

fn make_string_mask(series: &Series, target: &str) -> Series {
    let arr = series.data().as_any().downcast_ref::<StringArray>().unwrap();
    let mut b = BooleanArray::builder(arr.len());
    for i in 0..arr.len() {
        b.append_value(arr.is_valid(i) && arr.value(i) == target);
    }
    Series::new("mask", Arc::new(b.finish()))
}

fn count_csv_rows(path: &str) -> usize {
    let mut rdr = csv::Reader::from_path(path).unwrap();
    rdr.records().count()
}

fn read_csv_column(path: &str, col: usize) -> Vec<String> {
    let mut rdr = csv::Reader::from_path(path).unwrap();
    rdr.records().map(|r| r.unwrap().get(col).unwrap().to_string()).collect()
}

#[test]
fn test_save_senior_employees() {
    ensure_output_dir();
    let df = load_test_data();
    let mask = make_int_mask(df.select("age").unwrap(), |v| v >= 50);
    let seniors = df.filter_by_mask(&mask).unwrap();
    let out = format!("{}/senior_employees.csv", OUTPUT_DIR);
    write_csv(&seniors, &out).unwrap();

    assert!(Path::new(&out).exists());
    let rows = count_csv_rows(&out);
    assert!(rows > 0 && rows < 1_000_000);

    let ages: Vec<i32> = read_csv_column(&out, 2).into_iter().map(|s| s.parse().unwrap()).collect();
    for age in &ages {
        assert!(*age >= 50);
    }
}

#[test]
fn test_save_high_performers() {
    ensure_output_dir();
    let df = load_test_data();
    let mask = make_int_mask(df.select("yield").unwrap(), |v| v > 9000);
    let performers = df.filter_by_mask(&mask).unwrap();
    let out = format!("{}/high_performers.csv", OUTPUT_DIR);
    write_csv(&performers, &out).unwrap();

    assert!(Path::new(&out).exists());
    assert!(count_csv_rows(&out) > 0);

    let yields: Vec<i32> = read_csv_column(&out, 4).into_iter().map(|s| s.parse().unwrap()).collect();
    for y in &yields {
        assert!(*y > 9000);
    }
}

#[test]
fn test_save_managers() {
    ensure_output_dir();
    let df = load_test_data();
    let mask = make_string_mask(df.select("position").unwrap(), "Manager");
    let managers = df.filter_by_mask(&mask).unwrap();
    let out = format!("{}/managers.csv", OUTPUT_DIR);
    write_csv(&managers, &out).unwrap();

    assert!(Path::new(&out).exists());
    assert!(count_csv_rows(&out) > 0);

    for pos in &read_csv_column(&out, 3) {
        assert_eq!(pos, "Manager");
    }
}

#[test]
fn test_save_young_analysts() {
    ensure_output_dir();
    let df = load_test_data();
    let analysts = df.filter_by_mask(&make_string_mask(df.select("position").unwrap(), "Analyst")).unwrap();
    let young = analysts.filter_by_mask(&make_int_mask(analysts.select("age").unwrap(), |v| v < 30)).unwrap();
    let out = format!("{}/young_analysts.csv", OUTPUT_DIR);
    write_csv(&young, &out).unwrap();

    assert!(Path::new(&out).exists());
    assert!(count_csv_rows(&out) > 0);

    let ages: Vec<i32> = read_csv_column(&out, 2).into_iter().map(|s| s.parse().unwrap()).collect();
    let positions = read_csv_column(&out, 3);
    for (i, age) in ages.iter().enumerate() {
        assert!(*age < 30);
        assert_eq!(positions[i], "Analyst");
    }
}

#[test]
fn test_save_high_salary_directors() {
    ensure_output_dir();
    let df = load_test_data();
    let directors = df.filter_by_mask(&make_string_mask(df.select("position").unwrap(), "Director")).unwrap();
    let rich = directors.filter_by_mask(&make_float_mask(directors.select("salary").unwrap(), |v| v > 20000.0)).unwrap();
    let out = format!("{}/high_salary_directors.csv", OUTPUT_DIR);
    write_csv(&rich, &out).unwrap();

    assert!(Path::new(&out).exists());
    assert!(count_csv_rows(&out) > 0);

    let salaries: Vec<f64> = read_csv_column(&out, 1).into_iter().map(|s| s.parse().unwrap()).collect();
    let positions = read_csv_column(&out, 3);
    for (i, sal) in salaries.iter().enumerate() {
        assert!(*sal > 20000.0);
        assert_eq!(positions[i], "Director");
    }
}

#[test]
fn test_save_sorted_by_salary_desc() {
    ensure_output_dir();
    let df = load_test_data();
    let sorted = df.sort_by("salary", false).unwrap();
    let out = format!("{}/sorted_by_salary_desc.csv", OUTPUT_DIR);
    write_csv(&sorted, &out).unwrap();

    assert!(Path::new(&out).exists());
    assert_eq!(count_csv_rows(&out), 1_000_000);

    let salaries: Vec<f64> = read_csv_column(&out, 1).into_iter().map(|s| s.parse().unwrap()).collect();
    for i in 1..salaries.len() {
        assert!(salaries[i - 1] >= salaries[i]);
    }
}

#[test]
fn test_save_sorted_by_age_asc() {
    ensure_output_dir();
    let df = load_test_data();
    let sorted = df.sort_by("age", true).unwrap();
    let out = format!("{}/sorted_by_age_asc.csv", OUTPUT_DIR);
    write_csv(&sorted, &out).unwrap();

    assert!(Path::new(&out).exists());
    assert_eq!(count_csv_rows(&out), 1_000_000);

    let ages: Vec<i32> = read_csv_column(&out, 2).into_iter().map(|s| s.parse().unwrap()).collect();
    for i in 1..ages.len() {
        assert!(ages[i - 1] <= ages[i]);
    }
}

#[test]
fn test_save_grouped_position_counts() {
    ensure_output_dir();
    let df = load_test_data();
    let grouped = df.group_by("position").unwrap();
    let counts = grouped.count().unwrap();
    let out = format!("{}/position_counts.csv", OUTPUT_DIR);
    write_csv(&counts, &out).unwrap();

    assert!(Path::new(&out).exists());
    assert_eq!(count_csv_rows(&out), 5);
}

#[test]
fn test_save_grouped_salary_means() {
    ensure_output_dir();
    let df = load_test_data();
    let grouped = df.group_by("position").unwrap();
    let means = grouped.mean("salary").unwrap();
    let out = format!("{}/salary_means_by_position.csv", OUTPUT_DIR);
    write_csv(&means, &out).unwrap();

    assert!(Path::new(&out).exists());
    assert_eq!(count_csv_rows(&out), 5);

    for s in &read_csv_column(&out, 1).into_iter().map(|s| s.parse::<f64>().unwrap()).collect::<Vec<_>>() {
        assert!(*s > 0.0);
    }
}

#[test]
fn test_save_filtered_then_sorted() {
    ensure_output_dir();
    let df = load_test_data();
    let mask = make_int_mask(df.select("age").unwrap(), |v| v > 40);
    let over_40 = df.filter_by_mask(&mask).unwrap();
    let sorted = over_40.sort_by("salary", true).unwrap();
    let out = format!("{}/over_40_sorted_by_salary.csv", OUTPUT_DIR);
    write_csv(&sorted, &out).unwrap();

    assert!(Path::new(&out).exists());
    assert!(count_csv_rows(&out) > 0);

    let ages: Vec<i32> = read_csv_column(&out, 2).into_iter().map(|s| s.parse().unwrap()).collect();
    let salaries: Vec<f64> = read_csv_column(&out, 1).into_iter().map(|s| s.parse().unwrap()).collect();
    for age in &ages { assert!(*age > 40); }
    for i in 1..salaries.len() {
        assert!(salaries[i - 1] <= salaries[i]);
    }
}

#[test]
fn test_save_complex_combined_filter() {
    ensure_output_dir();
    let df = load_test_data();
    let engineers = df.filter_by_mask(&make_string_mask(df.select("position").unwrap(), "Engineer")).unwrap();
    let high_yield = engineers.filter_by_mask(&make_int_mask(engineers.select("yield").unwrap(), |v| v > 5000)).unwrap();
    let sorted = high_yield.sort_by("salary", false).unwrap();
    let out = format!("{}/engineers_high_yield_top.csv", OUTPUT_DIR);
    write_csv(&sorted, &out).unwrap();

    assert!(Path::new(&out).exists());
    assert!(count_csv_rows(&out) > 0);

    for pos in &read_csv_column(&out, 3) { assert_eq!(pos, "Engineer"); }
}

#[test]
fn test_save_empty_filter_result() {
    ensure_output_dir();
    let df = load_test_data();
    let all_false = Series::new("none", Arc::new(BooleanArray::from(vec![false; df.shape().0])));
    let empty = df.filter_by_mask(&all_false).unwrap();
    let out = format!("{}/empty_filter_result.csv", OUTPUT_DIR);
    write_csv(&empty, &out).unwrap();

    assert!(Path::new(&out).exists());
    assert_eq!(count_csv_rows(&out), 0);
}

#[test]
fn test_csv_roundtrip() {
    ensure_output_dir();
    let df = load_test_data();
    let out = format!("{}/roundtrip.csv", OUTPUT_DIR);
    write_csv(&df, &out).unwrap();
    let reloaded = read_csv(&out).unwrap();
    assert_eq!(df.shape(), reloaded.shape());
    assert_eq!(df.get_column_names(), reloaded.get_column_names());
    assert!(count_csv_rows(&out) > 0);
}

#[test]
fn test_parquet_roundtrip() {
    ensure_output_dir();
    let df = load_test_data();
    let out = format!("{}/roundtrip.parquet", OUTPUT_DIR);
    crossbow::io::parquet::write_parquet(&df, &out).unwrap();
    let reloaded = crossbow::io::parquet::read_parquet(&out).unwrap();
    assert_eq!(df.shape(), reloaded.shape());
    assert_eq!(df.get_column_names(), reloaded.get_column_names());
}
