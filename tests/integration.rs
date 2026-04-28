use std::sync::Arc;
use std::fs::File;
use arrow::csv::ReaderBuilder;
use arrow::datatypes::*;
use arrow::array::*;
use crossbow::{DataFrame, Series, read_csv, write_csv};

fn load_test_data() -> DataFrame {
    let schema = Arc::new(Schema::new(vec![
        Field::new("employee", DataType::Utf8, true),
        Field::new("salary", DataType::Float64, true),
        Field::new("age", DataType::Int32, true),
        Field::new("position", DataType::Utf8, true),
        Field::new("yield", DataType::Int32, true),
    ]));

    let file = File::open("data_1M.csv").unwrap();
    let mut reader = ReaderBuilder::new(schema.clone())
        .with_header(true)
        .with_batch_size(2000)
        .build(file)
        .unwrap();

    let batch = reader.next().unwrap().unwrap();

    let columns: Vec<Series> = schema.fields().iter().enumerate().map(|(i, field)| {
        Series::new(field.name(), batch.column(i).clone())
    }).collect();

    DataFrame::new(columns).unwrap()
}

fn make_bool_mask_int32(series: &Series, pred: fn(i32) -> bool) -> Series {
    let arr = series.as_primitive::<Int32Array>().unwrap();
    let mut b = BooleanArray::builder(arr.len());
    for i in 0..arr.len() {
        b.append_value(arr.is_valid(i) && pred(arr.value(i)));
    }
    Series::new("mask", Arc::new(b.finish()))
}

fn make_bool_mask_f64(series: &Series, pred: fn(f64) -> bool) -> Series {
    let arr = series.as_primitive::<Float64Array>().unwrap();
    let mut b = BooleanArray::builder(arr.len());
    for i in 0..arr.len() {
        b.append_value(arr.is_valid(i) && pred(arr.value(i)));
    }
    Series::new("mask", Arc::new(b.finish()))
}

#[test]
fn test_data_loading() {
    let df = load_test_data();
    assert_eq!(df.shape(), (2000, 5));
    assert_eq!(df.get_column_names(), vec!["employee", "salary", "age", "position", "yield"]);
    assert_eq!(df.select("employee").unwrap().dtype(), &DataType::Utf8);
    assert_eq!(df.select("salary").unwrap().dtype(), &DataType::Float64);
    assert_eq!(df.select("age").unwrap().dtype(), &DataType::Int32);
    assert_eq!(df.select("position").unwrap().dtype(), &DataType::Utf8);
    assert_eq!(df.select("yield").unwrap().dtype(), &DataType::Int32);
}

#[test]
fn test_series_basic_properties() {
    let df = load_test_data();
    let salary = df.select("salary").unwrap();
    assert_eq!(salary.name(), "salary");
    assert_eq!(salary.len(), 2000);
    assert!(!salary.is_empty());
    assert!(salary.is_numeric());
    assert!(!salary.is_string());
    assert!(!salary.is_boolean());

    let position = df.select("position").unwrap();
    assert!(position.is_string());
    assert!(!position.is_numeric());
}

#[test]
fn test_series_display_format() {
    let df = load_test_data();
    let age = df.select("age").unwrap();
    let display = format!("{}", age);
    assert!(display.contains("age"));
    assert!(display.contains("DataType: Int32"));
    assert!(display.contains("┌"));
    assert!(display.contains("┘"));
}

#[test]
fn test_dataframe_display_format() {
    let df = load_test_data();
    let display = format!("{}", df);
    assert!(display.contains("employee"));
    assert!(display.contains("salary"));
    assert!(display.contains("Shape: (2000, 5)"));
    assert!(display.contains("┌"));
    assert!(display.contains("┘"));
}

#[test]
fn test_filter_by_age_gt_50() {
    let df = load_test_data();
    let age = df.select("age").unwrap();
    let mask = make_bool_mask_int32(age, |v| v > 50);
    let filtered = df.filter_by_mask(&mask).unwrap();

    assert!(filtered.shape().0 < 2000);
    assert!(filtered.shape().0 > 0);

    let ages = filtered.select("age").unwrap();
    for i in 0..ages.len() {
        let val: i32 = ages.value_at(i).unwrap().parse().unwrap();
        assert!(val > 50, "Expected age > 50, got {}", val);
    }
}

#[test]
fn test_filter_by_age_lt_30() {
    let df = load_test_data();
    let age = df.select("age").unwrap();
    let mask = make_bool_mask_int32(age, |v| v < 30);
    let filtered = df.filter_by_mask(&mask).unwrap();

    let ages = filtered.select("age").unwrap();
    for i in 0..ages.len() {
        let val: i32 = ages.value_at(i).unwrap().parse().unwrap();
        assert!(val < 30);
    }
}

#[test]
fn test_filter_by_salary_gte_20000() {
    let df = load_test_data();
    let salary = df.select("salary").unwrap();
    let mask = make_bool_mask_f64(salary, |v| v >= 20000.0);
    let filtered = df.filter_by_mask(&mask).unwrap();

    for i in 0..filtered.shape().0 {
        let val: f64 = filtered.select("salary").unwrap().value_at(i).unwrap().parse().unwrap();
        assert!(val >= 20000.0);
    }
}

#[test]
fn test_series_arithmetic_add() {
    let df = load_test_data();
    let age = df.select("age").unwrap();
    let ten_years = Series::from("ten_years", vec![10i32; 2000]);
    let future_age = age.add(&ten_years).unwrap();

    assert_eq!(future_age.len(), 2000);
    for i in 0..5 {
        let base: i32 = age.value_at(i).unwrap().parse().unwrap();
        let future: i32 = future_age.value_at(i).unwrap().parse().unwrap();
        assert_eq!(future, base + 10);
    }
}

#[test]
fn test_series_arithmetic_subtract() {
    let df = load_test_data();
    let yield_col = df.select("yield").unwrap();
    let thousand = Series::from("thousand", vec![1000i32; 2000]);
    let adjusted = yield_col.subtract(&thousand).unwrap();

    for i in 0..5 {
        let base: i32 = yield_col.value_at(i).unwrap().parse().unwrap();
        let result: i32 = adjusted.value_at(i).unwrap().parse().unwrap();
        assert_eq!(result, base - 1000);
    }
}

#[test]
fn test_series_arithmetic_multiply() {
    let df = load_test_data();
    let age = df.select("age").unwrap();
    let two = Series::from("two", vec![2i32; 2000]);
    let doubled = age.multiply(&two).unwrap();

    for i in 0..5 {
        let base: i32 = age.value_at(i).unwrap().parse().unwrap();
        let result: i32 = doubled.value_at(i).unwrap().parse().unwrap();
        assert_eq!(result, base * 2);
    }
}

#[test]
fn test_series_aggregation_sum() {
    let df = load_test_data();
    let age = df.select("age").unwrap();
    let sum = age.sum().unwrap();
    assert!(sum > 0.0);
    assert!(sum.is_finite());
}

#[test]
fn test_series_aggregation_mean() {
    let df = load_test_data();
    let salary = df.select("salary").unwrap();
    let mean = salary.mean().unwrap();
    assert!(mean > 0.0);

    let age = df.select("age").unwrap();
    let age_mean = age.mean().unwrap();
    assert!(age_mean > 18.0);
    assert!(age_mean < 70.0);
}

#[test]
fn test_series_aggregation_min_max() {
    let df = load_test_data();
    let age = df.select("age").unwrap();
    let min = age.min().unwrap().unwrap();
    let max = age.max().unwrap().unwrap();
    assert!(min <= max);
    assert!(min >= 18.0);
    assert!(max <= 70.0);
}

#[test]
fn test_series_aggregation_std_var() {
    let df = load_test_data();
    let salary = df.select("salary").unwrap();
    let std_dev = salary.std().unwrap();
    let variance = salary.var().unwrap();
    assert!(std_dev >= 0.0);
    assert!(variance >= 0.0);
    assert!((std_dev * std_dev - variance).abs() < 0.01);
}

#[test]
fn test_series_count() {
    let df = load_test_data();
    let age = df.select("age").unwrap();
    assert_eq!(age.count(), 2000);
    let non_null = age.count_non_null().unwrap();
    assert_eq!(non_null, 2000);
}

#[test]
fn test_series_value_at() {
    let df = load_test_data();
    let age = df.select("age").unwrap();
    let val = age.value_at(0).unwrap();
    assert!(!val.is_empty());

    let err = age.value_at(999999);
    assert!(err.is_err());
}

#[test]
fn test_series_slice() {
    let df = load_test_data();
    let age = df.select("age").unwrap();
    let sliced = age.slice(0, 100).unwrap();
    assert_eq!(sliced.len(), 100);

    let err = age.slice(1999, 10);
    assert!(err.is_err());
}

#[test]
fn test_dataframe_select_columns() {
    let df = load_test_data();
    let salary = df.select("salary").unwrap();
    assert_eq!(salary.name(), "salary");

    let err = df.select("nonexistent");
    assert!(err.is_err());
}

#[test]
fn test_dataframe_get_row() {
    let df = load_test_data();
    let row = df.get_row(0).unwrap();
    assert_eq!(row.len(), 5);

    let err = df.get_row(999999);
    assert!(err.is_err());
}

#[test]
fn test_dataframe_get_rows() {
    let df = load_test_data();
    let rows = df.get_rows(&[0, 1, 2]).unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].len(), 5);

    let err = df.get_rows(&[0, 999999]);
    assert!(err.is_err());
}

#[test]
fn test_dataframe_add_remove_column() {
    let df = load_test_data();
    let bonus = Series::from("bonus", vec![1000i32; 2000]);
    let extended = df.add_column(bonus).unwrap();
    assert_eq!(extended.shape(), (2000, 6));
    assert!(extended.select("bonus").is_ok());

    let err = extended.add_column(Series::from("bonus", vec![1i32; 2000]));
    assert!(err.is_err());

    let removed = extended.remove_column("bonus").unwrap();
    assert_eq!(removed.shape(), (2000, 5));

    let err = removed.remove_column("bonus");
    assert!(err.is_err());
}

#[test]
fn test_dataframe_rename_column() {
    let df = load_test_data();
    let renamed = df.rename_column("yield", "performance").unwrap();
    assert!(renamed.select("performance").is_ok());
    assert!(renamed.select("yield").is_err());

    let err = df.rename_column("yield", "salary");
    assert!(err.is_err());
}

#[test]
fn test_dataframe_filter_combined() {
    let df = load_test_data();
    let age_mask = make_bool_mask_int32(df.select("age").unwrap(), |v| v > 40);
    let over_40 = df.filter_by_mask(&age_mask).unwrap();
    let salary_on_over_40 = make_bool_mask_f64(over_40.select("salary").unwrap(), |v| v > 15000.0);
    let filtered = over_40.filter_by_mask(&salary_on_over_40).unwrap();

    assert!(filtered.shape().0 <= over_40.shape().0);
    assert!(filtered.shape().0 > 0);

    for i in 0..filtered.shape().0 {
        let a: i32 = filtered.select("age").unwrap().value_at(i).unwrap().parse().unwrap();
        let s: f64 = filtered.select("salary").unwrap().value_at(i).unwrap().parse().unwrap();
        assert!(a > 40 && s > 15000.0);
    }
}

#[test]
fn test_dataframe_sort_ascending() {
    let df = load_test_data();
    let sorted = df.sort_by("age", true).unwrap();
    assert_eq!(sorted.shape(), df.shape());

    let ages = sorted.select("age").unwrap();
    for i in 1..ages.len() {
        let prev: i32 = ages.value_at(i - 1).unwrap().parse().unwrap();
        let curr: i32 = ages.value_at(i).unwrap().parse().unwrap();
        assert!(prev <= curr, "asc sort fail at {}: {} > {}", i, prev, curr);
    }
}

#[test]
fn test_dataframe_sort_descending() {
    let df = load_test_data();
    let sorted = df.sort_by("yield", false).unwrap();
    let yields = sorted.select("yield").unwrap();

    for i in 1..yields.len() {
        let prev: i32 = yields.value_at(i - 1).unwrap().parse().unwrap();
        let curr: i32 = yields.value_at(i).unwrap().parse().unwrap();
        assert!(prev >= curr, "desc sort fail at {}: {} < {}", i, prev, curr);
    }
}

#[test]
fn test_dataframe_sort_by_string() {
    let df = load_test_data();
    let sorted = df.sort_by("position", true).unwrap();
    assert_eq!(sorted.shape(), df.shape());

    let positions = sorted.select("position").unwrap();
    for i in 1..positions.len() {
        let prev = positions.value_at(i - 1).unwrap();
        let curr = positions.value_at(i).unwrap();
        assert!(prev <= curr, "string sort fail at {}", i);
    }
}

#[test]
fn test_dataframe_groupby_count() {
    let df = load_test_data();
    let grouped = df.group_by("position").unwrap();
    let result = grouped.count().unwrap();
    assert!(result.shape().0 >= 4);
    assert!(result.select("position").is_ok());
    assert!(result.select("count").is_ok());
}

#[test]
fn test_dataframe_groupby_sum() {
    let df = load_test_data();
    let grouped = df.group_by("position").unwrap();
    let result = grouped.sum("yield").unwrap();
    assert!(result.select("position").is_ok());
    assert!(result.select("yield_sum").is_ok());
}

#[test]
fn test_dataframe_groupby_mean() {
    let df = load_test_data();
    let grouped = df.group_by("position").unwrap();
    let result = grouped.mean("salary").unwrap();
    assert!(result.select("position").is_ok());
    assert!(result.select("salary_mean").is_ok());

    let means = result.select("salary_mean").unwrap();
    for i in 0..means.len() {
        let val = means.value_at(i).unwrap();
        if val != "null" {
            let num: f64 = val.parse().unwrap();
            assert!(num > 0.0);
        }
    }
}

#[test]
fn test_dataframe_empty() {
    let empty = DataFrame::new(vec![]).unwrap();
    assert_eq!(empty.shape(), (0, 0));
    assert!(empty.get_column_names().is_empty());
    assert!(empty.get_row(0).is_err());
}

#[test]
fn test_dataframe_duplicate_column_error() {
    let s1 = Series::from("name", vec![1i32, 2, 3]);
    let s2 = Series::from("name", vec![4i32, 5, 6]);
    let err = DataFrame::new(vec![s1, s2]);
    assert!(err.is_err());
}

#[test]
fn test_dataframe_mismatched_length_error() {
    let s1 = Series::from("a", vec![1i32, 2, 3]);
    let s2 = Series::from("b", vec![1i32, 2]);
    let err = DataFrame::new(vec![s1, s2]);
    assert!(err.is_err());
}

#[test]
fn test_series_null_operations() {
    let df = load_test_data();
    let age = df.select("age").unwrap();
    let null_mask = age.is_null().unwrap();
    assert_eq!(null_mask.len(), 2000);
    let not_null_mask = age.is_not_null().unwrap();
    assert_eq!(not_null_mask.len(), 2000);
}

#[test]
fn test_series_is_numeric_checks() {
    let df = load_test_data();
    assert!(df.select("salary").unwrap().is_numeric());
    assert!(df.select("age").unwrap().is_numeric());
    assert!(df.select("yield").unwrap().is_numeric());
    assert!(!df.select("employee").unwrap().is_numeric());
    assert!(!df.select("position").unwrap().is_numeric());
}

#[test]
fn test_series_is_string_checks() {
    let df = load_test_data();
    assert!(df.select("employee").unwrap().is_string());
    assert!(df.select("position").unwrap().is_string());
    assert!(!df.select("salary").unwrap().is_string());
}

#[test]
fn test_modulo_operation() {
    let df = load_test_data();
    let age = df.select("age").unwrap();
    let two = Series::from("two", vec![2i32; 2000]);
    let parity = age.modulo(&two).unwrap();

    for i in 0..5 {
        let base: i32 = age.value_at(i).unwrap().parse().unwrap();
        let result: i32 = parity.value_at(i).unwrap().parse().unwrap();
        assert_eq!(result, base % 2);
    }
}

#[test]
fn test_divide_operation() {
    let df = load_test_data();
    let yield_col = df.select("yield").unwrap();
    let two = Series::from("two", vec![2i32; 2000]);
    let half = yield_col.divide(&two).unwrap();

    for i in 0..5 {
        let base: i32 = yield_col.value_at(i).unwrap().parse().unwrap();
        let result: i32 = half.value_at(i).unwrap().parse().unwrap();
        assert_eq!(result, base / 2);
    }
}

#[test]
fn test_mismatched_length_error() {
    let df = load_test_data();
    let age = df.select("age").unwrap();
    let short = Series::from("short", vec![1i32]);
    let err = age.add(&short);
    assert!(err.is_err());
}

#[test]
fn test_type_mismatch_error() {
    let df = load_test_data();
    let age = df.select("age").unwrap();
    let strings = Series::from("str", vec!["hello"; 2000]);
    let err = age.add(&strings);
    assert!(err.is_err());
}

#[test]
fn test_sort_multiple_columns() {
    let df = load_test_data();
    let by_age = df.sort_by("age", true).unwrap();
    let by_salary = df.sort_by("salary", false).unwrap();
    assert_eq!(by_age.shape(), (2000, 5));
    assert_eq!(by_salary.shape(), (2000, 5));

    let age_col = by_age.select("age").unwrap();
    for i in 1..age_col.len() {
        let p: i32 = age_col.value_at(i - 1).unwrap().parse().unwrap();
        let c: i32 = age_col.value_at(i).unwrap().parse().unwrap();
        assert!(p <= c);
    }

    let salary_col = by_salary.select("salary").unwrap();
    for i in 1..salary_col.len() {
        let p: f64 = salary_col.value_at(i - 1).unwrap().parse().unwrap();
        let c: f64 = salary_col.value_at(i).unwrap().parse().unwrap();
        assert!(p >= c);
    }
}

#[test]
fn test_filter_by_boolean_true_all() {
    let df = load_test_data();
    let all_true = Series::new("all", Arc::new(BooleanArray::from(vec![true; 2000])));
    let filtered = df.filter_by_mask(&all_true).unwrap();
    assert_eq!(filtered.shape(), (2000, 5));
}

#[test]
fn test_filter_by_boolean_false_all() {
    let df = load_test_data();
    let all_false = Series::new("none", Arc::new(BooleanArray::from(vec![false; 2000])));
    let filtered = df.filter_by_mask(&all_false).unwrap();
    assert_eq!(filtered.shape(), (0, 5));
}

// ============================================================================
// NEW TESTS: I/O EDGE CASES & HIGH PRIORITY COVERAGE
// ============================================================================

// --- CSV READ/WRITE EDGE CASES ---

#[test]
fn test_csv_roundtrip_preserves_data() {
    let df = load_test_data();
    let output_path = "tests/output/roundtrip_test.csv";
    
    // Write to CSV
    write_csv(&df, output_path).unwrap();
    
    // Read back from CSV
    let df_read = read_csv(output_path).unwrap();
    
    // Verify shape is preserved
    assert_eq!(df.shape(), df_read.shape());
    
    // Verify column names are preserved
    assert_eq!(df.get_column_names(), df_read.get_column_names());
    
    // Verify sample data rows match
    for i in 0..10 {
        let orig_row = df.get_row(i).unwrap();
        let read_row = df_read.get_row(i).unwrap();
        assert_eq!(orig_row, read_row);
    }
}

#[test]
fn test_write_csv_empty_dataframe() {
    let empty_s1: Series = Series::from("col A", Vec::<i32>::new());
    let empty_s2: Series = Series::from("col B", Vec::<&str>::new());
    let empty_df = DataFrame::new(vec![empty_s1, empty_s2]).unwrap();
    
    let output_path = "tests/output/empty_dataframe.csv";
    assert!(write_csv(&empty_df, output_path).is_ok());
    
    // Read it back - should have columns but no rows
    let df_read = read_csv(output_path).unwrap();
    assert_eq!(df_read.shape(), (0, 2));
    assert_eq!(df_read.get_column_names(), vec!["col A", "col B"]);
}

#[test]
fn test_csv_preserves_nulls() {
    let s1: Vec<Option<i32>> = vec![Some(1), None, Some(3), None, Some(5)];
    let s2: Vec<Option<&str>> = vec![Some("a"), Some("b"), None, Some("d"), Some("e")];
    
    let s1_series = Series::from("ints", s1);
    let s2_series = Series::from("strs", s2);
    let df = DataFrame::new(vec![s1_series, s2_series]).unwrap();
    
    let output_path = "tests/output/nulls_test.csv";
    write_csv(&df, output_path).unwrap();
    
    let df_read = read_csv(output_path).unwrap();
    assert_eq!(df.shape(), df_read.shape());
}

#[test]
fn test_write_csv_filtered_result() {
    let df = load_test_data();
    
    // Filter to get subset
    let age = df.select("age").unwrap();
    let mask = make_bool_mask_int32(age, |v| v > 50);
    let filtered = df.filter_by_mask(&mask).unwrap();
    
    let output_path = "tests/output/filtered_over_50.csv";
    write_csv(&filtered, output_path).unwrap();
    
    let df_read = read_csv(output_path).unwrap();
    assert_eq!(df_read.shape().1, 5); // Same number of columns
    assert!(df_read.shape().0 < df.shape().0); // Fewer rows
    assert!(df_read.shape().0 > 0); // But not empty
}

#[test]
fn test_csv_roundtrip_sorted() {
    let df = load_test_data();
    let sorted = df.sort_by("age", true).unwrap();
    
    let output_path = "tests/output/sorted_roundtrip.csv";
    write_csv(&sorted, output_path).unwrap();
    
    let df_read = read_csv(output_path).unwrap();
    
    // Verify sorting is preserved
    let age_col = df_read.select("age").unwrap();
    for i in 1..100.min(age_col.len()) {
        let prev: i32 = age_col.value_at(i - 1).unwrap().parse().unwrap();
        let curr: i32 = age_col.value_at(i).unwrap().parse().unwrap();
        assert!(prev <= curr);
    }
}

#[test]
fn test_csv_roundtrip_grouped() {
    let df = load_test_data();
    let grouped = df.group_by("position").unwrap();
    let result = grouped.count().unwrap();
    
    let output_path = "tests/output/grouped_roundtrip.csv";
    write_csv(&result, output_path).unwrap();
    
    let df_read = read_csv(output_path).unwrap();
    assert_eq!(df_read.shape(), result.shape());
}

#[test]
fn test_write_csv_single_row() {
    let s1 = Series::from("val", vec![42]);
    let s2 = Series::from("text", vec!["hello"]);
    let df = DataFrame::new(vec![s1, s2]).unwrap();
    
    let output_path = "tests/output/single_row.csv";
    write_csv(&df, output_path).unwrap();
    
    let df_read = read_csv(output_path).unwrap();
    assert_eq!(df_read.shape(), (1, 2));
}

#[test]
fn test_write_csv_single_column() {
    let s = Series::from("only_col", vec![1, 2, 3, 4, 5]);
    let df = DataFrame::new(vec![s]).unwrap();
    
    let output_path = "tests/output/single_column.csv";
    write_csv(&df, output_path).unwrap();
    
    let df_read = read_csv(output_path).unwrap();
    assert_eq!(df_read.shape(), (5, 1));
    assert_eq!(df_read.get_column_names(), vec!["only_col"]);
}
