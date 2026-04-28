use std::fs::{self, File};
use std::sync::Arc;
use std::path::Path;
use arrow::csv::ReaderBuilder;
use arrow::datatypes::*;
use arrow::array::*;
use arrow::compute::concat_batches;
use crossbow::{DataFrame, Series};

static OUTPUT_DIR: &str = "tests/output";

fn ensure_output_dir() {
    let _ = fs::create_dir_all(OUTPUT_DIR);
}

fn load_test_data() -> DataFrame {
    let schema = Arc::new(Schema::new(vec![
        Field::new("employee", DataType::Utf8, true),
        Field::new("salary", DataType::Float64, true),
        Field::new("age", DataType::Int32, true),
        Field::new("position", DataType::Utf8, true),
        Field::new("yield", DataType::Int32, true),
    ]));

    let file = File::open("data_1M.csv").unwrap();
    let reader = ReaderBuilder::new(schema.clone())
        .with_header(true)
        .with_batch_size(65536)
        .build(file)
        .unwrap();

    let batches: Vec<RecordBatch> = reader.collect::<Result<_, _>>().unwrap();
    let batch = concat_batches(&schema, batches.iter()).unwrap();

    let columns: Vec<Series> = schema.fields().iter().enumerate().map(|(i, field)| {
        Series::new(field.name(), batch.column(i).clone())
    }).collect();

    DataFrame::new(columns).unwrap()
}

fn make_int_mask(series: &Series, pred: fn(i32) -> bool) -> Series {
    let arr = series.as_primitive::<Int32Array>().unwrap();
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

fn write_df_to_csv(df: &DataFrame, path: &str) {
    let mut wtr = csv::Writer::from_path(path).unwrap();
    wtr.write_record(df.get_column_names()).unwrap();
    for i in 0..df.shape().0 {
        let row: Vec<String> = df.get_row(i).unwrap().into_iter().map(|v| {
            if v.starts_with('"') && v.ends_with('"') && v.len() >= 2 {
                v[1..v.len() - 1].to_string()
            } else {
                v
            }
        }).collect();
        wtr.write_record(&row).unwrap();
    }
    wtr.flush().unwrap();
}

fn count_csv_rows(path: &str) -> usize {
    let mut rdr = csv::Reader::from_path(path).unwrap();
    rdr.records().count()
}

fn read_csv_column(path: &str, col: usize) -> Vec<String> {
    let mut rdr = csv::Reader::from_path(path).unwrap();
    rdr.records().map(|r| r.unwrap().get(col).unwrap().to_string()).collect()
}

// ---- Filter and save tests ----

#[test]
fn test_save_senior_employees() {
    ensure_output_dir();
    let df = load_test_data();
    let mask = make_int_mask(df.select("age").unwrap(), |v| v >= 50);
    let seniors = df.filter_by_mask(&mask).unwrap();
    let out = format!("{}/senior_employees.csv", OUTPUT_DIR);
    write_df_to_csv(&seniors, &out);

    assert!(Path::new(&out).exists());
    let rows = count_csv_rows(&out);
    assert!(rows > 0 && rows < 1_000_000);

    let ages: Vec<i32> = read_csv_column(&out, 2).into_iter().map(|s| s.parse().unwrap()).collect();
    for age in &ages {
        assert!(*age >= 50, "Expected age >= 50, got {}", age);
    }
}

#[test]
fn test_save_high_performers() {
    ensure_output_dir();
    let df = load_test_data();
    let mask = make_int_mask(df.select("yield").unwrap(), |v| v > 9000);
    let performers = df.filter_by_mask(&mask).unwrap();
    let out = format!("{}/high_performers.csv", OUTPUT_DIR);
    write_df_to_csv(&performers, &out);

    assert!(Path::new(&out).exists());
    let rows = count_csv_rows(&out);
    assert!(rows > 0);

    let yields: Vec<i32> = read_csv_column(&out, 4).into_iter().map(|s| s.parse().unwrap()).collect();
    for y in &yields {
        assert!(*y > 9000, "Expected yield > 9000, got {}", y);
    }
}

#[test]
fn test_save_managers() {
    ensure_output_dir();
    let df = load_test_data();
    let mask = make_string_mask(df.select("position").unwrap(), "Manager");
    let managers = df.filter_by_mask(&mask).unwrap();
    let out = format!("{}/managers.csv", OUTPUT_DIR);
    write_df_to_csv(&managers, &out);

    assert!(Path::new(&out).exists());
    let rows = count_csv_rows(&out);
    assert!(rows > 0);

    let positions = read_csv_column(&out, 3);
    for pos in &positions {
        assert_eq!(pos, "Manager", "Expected position 'Manager', got '{}'", pos);
    }
}

#[test]
fn test_save_young_analysts() {
    ensure_output_dir();
    let df = load_test_data();
    let analysts = df.filter_by_mask(&make_string_mask(df.select("position").unwrap(), "Analyst")).unwrap();
    let young = analysts.filter_by_mask(&make_int_mask(analysts.select("age").unwrap(), |v| v < 30)).unwrap();
    let out = format!("{}/young_analysts.csv", OUTPUT_DIR);
    write_df_to_csv(&young, &out);

    assert!(Path::new(&out).exists());
    assert!(count_csv_rows(&out) > 0);

    let ages: Vec<i32> = read_csv_column(&out, 2).into_iter().map(|s| s.parse().unwrap()).collect();
    let positions = read_csv_column(&out, 3);
    for (i, age) in ages.iter().enumerate() {
        assert!(*age < 30, "Expected age < 30, got {}", age);
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
    write_df_to_csv(&rich, &out);

    assert!(Path::new(&out).exists());
    assert!(count_csv_rows(&out) > 0);

    let salaries: Vec<f64> = read_csv_column(&out, 1).into_iter().map(|s| s.parse().unwrap()).collect();
    let positions = read_csv_column(&out, 3);
    for (i, sal) in salaries.iter().enumerate() {
        assert!(*sal > 20000.0, "Expected salary > 20000, got {}", sal);
        assert_eq!(positions[i], "Director");
    }
}

#[test]
fn test_save_sorted_by_salary_desc() {
    ensure_output_dir();
    let df = load_test_data();
    let sorted = df.sort_by("salary", false).unwrap();
    let out = format!("{}/sorted_by_salary_desc.csv", OUTPUT_DIR);
    write_df_to_csv(&sorted, &out);

    assert!(Path::new(&out).exists());
    assert_eq!(count_csv_rows(&out), 1_000_000);

    let salaries: Vec<f64> = read_csv_column(&out, 1).into_iter().map(|s| s.parse().unwrap()).collect();
    for i in 1..salaries.len() {
        assert!(salaries[i - 1] >= salaries[i], "Sort failed at index {}: {} < {}", i, salaries[i - 1], salaries[i]);
    }
}

#[test]
fn test_save_sorted_by_age_asc() {
    ensure_output_dir();
    let df = load_test_data();
    let sorted = df.sort_by("age", true).unwrap();
    let out = format!("{}/sorted_by_age_asc.csv", OUTPUT_DIR);
    write_df_to_csv(&sorted, &out);

    assert!(Path::new(&out).exists());
    assert_eq!(count_csv_rows(&out), 1_000_000);

    let ages: Vec<i32> = read_csv_column(&out, 2).into_iter().map(|s| s.parse().unwrap()).collect();
    for i in 1..ages.len() {
        assert!(ages[i - 1] <= ages[i], "Sort failed at index {}: {} > {}", i, ages[i - 1], ages[i]);
    }
}

#[test]
fn test_save_grouped_position_counts() {
    ensure_output_dir();
    let df = load_test_data();
    let grouped = df.group_by("position").unwrap();
    let counts = grouped.count().unwrap();
    let out = format!("{}/position_counts.csv", OUTPUT_DIR);
    write_df_to_csv(&counts, &out);

    assert!(Path::new(&out).exists());
    let rows = count_csv_rows(&out);
    assert_eq!(rows, 5);
}

#[test]
fn test_save_grouped_salary_means() {
    ensure_output_dir();
    let df = load_test_data();
    let grouped = df.group_by("position").unwrap();
    let means = grouped.mean("salary").unwrap();
    let out = format!("{}/salary_means_by_position.csv", OUTPUT_DIR);
    write_df_to_csv(&means, &out);

    assert!(Path::new(&out).exists());
    assert_eq!(count_csv_rows(&out), 5);

    let salaries: Vec<f64> = read_csv_column(&out, 1).into_iter().map(|s| s.parse().unwrap()).collect();
    for s in &salaries {
        assert!(*s > 0.0, "Expected positive mean salary");
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
    write_df_to_csv(&sorted, &out);

    assert!(Path::new(&out).exists());
    assert!(count_csv_rows(&out) > 0);

    let ages: Vec<i32> = read_csv_column(&out, 2).into_iter().map(|s| s.parse().unwrap()).collect();
    let salaries: Vec<f64> = read_csv_column(&out, 1).into_iter().map(|s| s.parse().unwrap()).collect();
    for age in &ages { assert!(*age > 40); }
    for i in 1..salaries.len() {
        assert!(salaries[i - 1] <= salaries[i], "Sort failed at {}", i);
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
    write_df_to_csv(&sorted, &out);

    assert!(Path::new(&out).exists());
    assert!(count_csv_rows(&out) > 0);

    let positions = read_csv_column(&out, 3);
    for pos in &positions { assert_eq!(pos, "Engineer"); }
}

#[test]
fn test_save_empty_filter_result() {
    ensure_output_dir();
    let df = load_test_data();
    let all_false = Series::new("none", Arc::new(BooleanArray::from(vec![false; df.shape().0])));
    let empty = df.filter_by_mask(&all_false).unwrap();
    let out = format!("{}/empty_filter_result.csv", OUTPUT_DIR);
    write_df_to_csv(&empty, &out);

    assert!(Path::new(&out).exists());
    assert_eq!(count_csv_rows(&out), 0);
}


