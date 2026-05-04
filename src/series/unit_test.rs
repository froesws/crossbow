//! MIT License
//!
//! Copyright (c) [2025] [William Froes]
//! Project: [crossbow]
//! Developed at: [Uergs -- Universidade Estadual do Rio Grande do Sul]
use crate::series::Series;
use arrow::array::{Float64Array, Int32Array};
use arrow::datatypes::DataType;

#[test]
fn test_series_new() {
    let int_data: arrow::array::PrimitiveArray<arrow::datatypes::Int32Type> =
        arrow::array::PrimitiveArray::from(vec![Some(1), Some(2), Some(3)]);
    let series = Series::new("numbers", std::sync::Arc::new(int_data));

    assert_eq!(series.name(), "numbers");
    assert_eq!(series.len(), 3);
    assert_eq!(series.dtype(), &DataType::Int32);
}

#[test]
fn test_series_from_i32() {
    let series = Series::from("my_integers", vec![10, 20, 30]);
    assert_eq!(series.name(), "my_integers");
    assert_eq!(series.len(), 3);
    assert_eq!(series.dtype(), &DataType::Int32);
}

#[test]
fn test_series_from_optional_i32() {
    let series = Series::from("my_optional_integers", vec![Some(10), None, Some(30)]);
    assert_eq!(series.name(), "my_optional_integers");
    assert_eq!(series.len(), 3);
    assert_eq!(series.dtype(), &DataType::Int32);
    assert!(series.data().is_null(1));
    assert!(!series.data().is_valid(1));
}

#[test]
fn test_series_from_f64() {
    let series = Series::from("my_floats", vec![1.1, 2.2, 3.3]);
    assert_eq!(series.name(), "my_floats");
    assert_eq!(series.len(), 3);
    assert_eq!(series.dtype(), &DataType::Float64);
}

#[test]
fn test_series_from_bool() {
    let series = Series::from("my_booleans", vec![true, false, true]);
    assert_eq!(series.name(), "my_booleans");
    assert_eq!(series.len(), 3);
    assert_eq!(series.dtype(), &DataType::Boolean);
}

#[test]
fn test_series_from_str() {
    let series = Series::from("fruits", vec!["apple", "banana", "cherry"]);
    assert_eq!(series.name(), "fruits");
    assert_eq!(series.len(), 3);
    assert_eq!(series.dtype(), &DataType::Utf8);
}

#[test]
fn test_series_from_optional_str() {
    let series = Series::from("optional_fruits", vec![Some("apple"), None, Some("cherry")]);
    assert_eq!(series.name(), "optional_fruits");
    assert_eq!(series.len(), 3);
    assert_eq!(series.dtype(), &DataType::Utf8);
    assert!(series.data().is_null(1));
}

#[test]
fn test_series_from_string() {
    let data = vec!["apple".to_string(), "banana".to_string()];
    let series = Series::from("string_fruits", data);
    assert_eq!(series.name(), "string_fruits");
    assert_eq!(series.len(), 2);
    assert_eq!(series.dtype(), &DataType::Utf8);
}

#[test]
fn test_as_primitive_downcast() {
    let series_i32 = Series::from("numbers", vec![1, 2, 3]);

    if let Some(int_array) = series_i32.as_primitive::<Int32Array>() {
        assert_eq!(int_array.value(0), 1);
    } else {
        panic!("Downcast to Int32Array should succeed");
    }

    let float_array = series_i32.as_primitive::<Float64Array>();
    assert!(
        float_array.is_none(),
        "Downcast to Float64Array should fail"
    );
}

#[test]
fn test_get_value_as_string() {
    let series_i32 = Series::from("test_i32", vec![Some(123), None]);
    assert_eq!(series_i32.get_value_as_string(0), "123");
    assert_eq!(series_i32.get_value_as_string(1), "null");

    let series_f64 = Series::from("test_f64", vec![Some(45.6), None]);
    assert_eq!(series_f64.get_value_as_string(0), "45.6");
    assert_eq!(series_f64.get_value_as_string(1), "null");

    let series_bool = Series::from("test_bool", vec![Some(true), None]);
    assert_eq!(series_bool.get_value_as_string(0), "true");
    assert_eq!(series_bool.get_value_as_string(1), "null");

    let series_str = Series::from("test_str", vec![Some("hello"), None]);
    assert_eq!(series_str.get_value_as_string(0), "\"hello\"");
    assert_eq!(series_str.get_value_as_string(1), "null");
}

#[test]
#[should_panic]
fn test_get_value_as_string_out_of_bounds() {
    let series = Series::from("panic_test", vec![1, 2]);
    series.get_value_as_string(99);
}

#[test]
fn test_series_len() {
    let series = Series::from("length_test", vec![1, 2, 3, 4, 5]);
    assert_eq!(series.len(), 5);
}

// ============================================================================
// NEW TESTS: EDGE CASES & HIGH PRIORITY COVERAGE
// ============================================================================

// --- NULL HANDLING EDGE CASES ---

#[test]
fn test_series_all_null_arithmetic_add() {
    let all_null: Vec<Option<i32>> = vec![None, None, None];
    let series_null = Series::from("null_series", all_null);
    let series_normal = Series::from("normal", vec![1, 2, 3]);

    let result = series_null.add(&series_normal);
    assert!(result.is_ok());
    let result = result.unwrap();
    assert_eq!(result.len(), 3);
    for i in 0..3 {
        assert_eq!(result.get_value_as_string(i), "null");
    }
}

#[test]
fn test_series_all_null_arithmetic_multiply() {
    let all_null: Vec<Option<i32>> = vec![None, None, None];
    let series_null = Series::from("null_series", all_null);
    let series_normal = Series::from("normal", vec![5, 10, 15]);

    let result = series_null.multiply(&series_normal);
    assert!(result.is_ok());
    let result = result.unwrap();
    for i in 0..3 {
        assert_eq!(result.get_value_as_string(i), "null");
    }
}

#[test]
fn test_series_all_null_sum() {
    let all_null: Vec<Option<i32>> = vec![None, None, None];
    let series_null = Series::from("null_series", all_null);

    let sum = series_null.sum();
    assert!(sum.is_ok());
    assert_eq!(sum.unwrap(), 0.0);
}

#[test]
fn test_series_all_null_mean() {
    let all_null: Vec<Option<i32>> = vec![None, None, None];
    let series_null = Series::from("null_series", all_null);

    let mean = series_null.mean();
    // Mean of all-null series should either be NaN or 0.0
    assert!(mean.is_ok());
    let mean_val = mean.unwrap();
    assert!(mean_val.is_nan() || mean_val == 0.0);
}

#[test]
fn test_series_all_null_min() {
    let all_null: Vec<Option<i32>> = vec![None, None, None];
    let series_null = Series::from("null_series", all_null);

    let min = series_null.min();
    assert!(min.is_ok());
    assert!(min.unwrap().is_none());
}

#[test]
fn test_series_mixed_null_arithmetic() {
    let series1: Vec<Option<i32>> = vec![Some(1), None, Some(3)];
    let series2: Vec<Option<i32>> = vec![Some(10), Some(20), None];

    let s1 = Series::from("s1", series1);
    let s2 = Series::from("s2", series2);

    let result = s1.add(&s2).unwrap();
    assert_eq!(result.get_value_as_string(0), "11");
    assert_eq!(result.get_value_as_string(1), "null");
    assert_eq!(result.get_value_as_string(2), "null");
}

// --- DIVISION & MODULO EDGE CASES ---

#[test]
fn test_series_division_by_zero_error() {
    let series = Series::from("nums", vec![10, 20, 30]);
    let zeros = Series::from("zeros", vec![1, 0, 1]);

    let result = series.divide(&zeros);
    assert!(result.is_err());
}

#[test]
fn test_series_modulo_by_zero_error() {
    let series = Series::from("nums", vec![10, 20, 30]);
    let zeros = Series::from("zeros", vec![1, 0, 1]);

    let result = series.modulo(&zeros);
    assert!(result.is_err());
}

#[test]
fn test_series_modulo_negative_numbers() {
    let series = Series::from("nums", vec![-5, -3, 3, 5]);
    let divisor = Series::from("div", vec![2, 2, 2, 2]);

    let result = series.modulo(&divisor).unwrap();
    assert_eq!(result.get_value_as_string(0), "-1");
    assert_eq!(result.get_value_as_string(1), "-1");
    assert_eq!(result.get_value_as_string(2), "1");
    assert_eq!(result.get_value_as_string(3), "1");
}

// --- BOUNDARY CONDITIONS ---

#[test]
fn test_series_slice_exact_boundaries() {
    let series = Series::from("nums", vec![1, 2, 3, 4, 5]);

    // Slice entire series
    let full = series.slice(0, 5).unwrap();
    assert_eq!(full.len(), 5);

    // Slice single element
    let single = series.slice(2, 1).unwrap();
    assert_eq!(single.len(), 1);
    assert_eq!(single.get_value_as_string(0), "3");

    // Slice at end boundary
    let at_end = series.slice(4, 1).unwrap();
    assert_eq!(at_end.len(), 1);
    assert_eq!(at_end.get_value_as_string(0), "5");
}

#[test]
fn test_series_slice_out_of_bounds() {
    let series = Series::from("nums", vec![1, 2, 3]);

    // Start beyond length
    let result = series.slice(5, 1);
    assert!(result.is_err());

    // End beyond length
    let result = series.slice(0, 100);
    assert!(result.is_err());

    // Start + length beyond length
    let result = series.slice(2, 5);
    assert!(result.is_err());
}

#[test]
fn test_empty_series_operations() {
    let empty: Vec<i32> = vec![];
    let series = Series::from("empty", empty);

    assert_eq!(series.len(), 0);
    assert!(series.is_empty());

    // Aggregations on empty series
    assert_eq!(series.sum().unwrap(), 0.0);
    let mean = series.mean().unwrap();
    assert!(mean.is_nan() || mean == 0.0);
    assert!(series.min().unwrap().is_none());
    assert_eq!(series.count(), 0);
}

#[test]
fn test_empty_series_slice() {
    let empty: Vec<i32> = vec![];
    let series = Series::from("empty", empty);

    let result = series.slice(0, 0);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 0);
}

// --- TYPE CONVERSIONS & SAFETY ---

#[test]
fn test_series_downcast_wrong_type_returns_none() {
    let series_i32 = Series::from("nums", vec![1, 2, 3]);
    assert!(
        series_i32
            .as_primitive::<arrow::array::Float64Array>()
            .is_none()
    );

    let series_f64 = Series::from("floats", vec![1.1, 2.2, 3.3]);
    assert!(
        series_f64
            .as_primitive::<arrow::array::Int32Array>()
            .is_none()
    );
}

#[test]
fn test_series_string_value_at_whitespace() {
    let series = Series::from("strings", vec![Some("  hello  "), Some(""), None]);
    assert_eq!(series.get_value_as_string(0), "\"  hello  \"");
    assert_eq!(series.get_value_as_string(1), "\"\"");
    assert_eq!(series.get_value_as_string(2), "null");
}

// --- EXTREME VALUES ---

#[test]
fn test_series_extreme_i32_values() {
    let series = Series::from("extreme", vec![i32::MAX, i32::MIN, 0, -1]);
    assert!(series.get_value_as_string(0).parse::<i32>().is_ok());
    assert_eq!(
        series.get_value_as_string(0).parse::<i32>().unwrap(),
        i32::MAX
    );
    assert_eq!(
        series.get_value_as_string(1).parse::<i32>().unwrap(),
        i32::MIN
    );
}

#[test]
fn test_series_arithmetic_near_overflow() {
    let series = Series::from("nums", vec![i32::MAX - 1, i32::MAX - 1]);
    let one = Series::from("one", vec![1, 1]);

    let result = series.add(&one);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().get_value_as_string(0), i32::MAX.to_string());
}

#[test]
fn test_series_float_special_values() {
    let series = Series::from("floats", vec![f64::INFINITY, f64::NEG_INFINITY, 0.0]);
    assert!(
        series.get_value_as_string(0).contains("inf")
            || series.get_value_as_string(0).contains("∞")
    );
    assert_eq!(series.get_value_as_string(2), "0");
}

// --- SINGLE ELEMENT OPERATIONS ---

#[test]
fn test_single_element_series() {
    let series = Series::from("single", vec![42]);
    assert_eq!(series.len(), 1);
    assert_eq!(series.sum().unwrap(), 42.0);
    assert_eq!(series.mean().unwrap(), 42.0);
}

#[test]
fn test_single_element_arithmetic() {
    let s1 = Series::from("s1", vec![10]);
    let s2 = Series::from("s2", vec![5]);

    let sum = s1.add(&s2).unwrap();
    assert_eq!(sum.get_value_as_string(0), "15");

    let product = s1.multiply(&s2).unwrap();
    assert_eq!(product.get_value_as_string(0), "50");
}

#[test]
fn test_mean_all_null_returns_nan() {
    let all_null: Vec<Option<i32>> = vec![None, None, None];
    let series = Series::from("null_series", all_null);
    let mean = series.mean().unwrap();
    assert!(mean.is_nan(), "Expected NaN, got {}", mean);
}

#[test]
fn test_mean_empty_series_returns_nan() {
    let empty: Vec<i32> = vec![];
    let series = Series::from("empty", empty);
    let mean = series.mean().unwrap();
    assert!(mean.is_nan(), "Expected NaN for empty series, got {}", mean);
}

#[test]
fn test_std_single_value_returns_nan() {
    let single = Series::from("single", vec![42i32]);
    let std_val = single.std().unwrap();
    assert!(
        std_val.is_nan(),
        "Expected NaN for single value std, got {}",
        std_val
    );
}

#[test]
fn test_std_two_values_returns_valid() {
    let two = Series::from("two", vec![10i32, 20]);
    let std_val = two.std().unwrap();
    assert!(!std_val.is_nan());
    assert!(std_val > 0.0);
}

#[test]
fn test_var_all_null_returns_nan() {
    let all_null: Vec<Option<i32>> = vec![None, None, None];
    let series = Series::from("null_series", all_null);
    let var_val = series.var().unwrap();
    assert!(
        var_val.is_nan(),
        "Expected NaN for all-null var, got {}",
        var_val
    );
}

#[test]
fn test_var_single_value_returns_nan() {
    let single = Series::from("single", vec![42i32]);
    let var_val = single.var().unwrap();
    assert!(
        var_val.is_nan(),
        "Expected NaN for single value var, got {}",
        var_val
    );
}

// ============================================================================
// DATE TYPE TESTS
// ============================================================================

#[test]
fn test_date32_construction_and_display() {
    // 2023-12-10 = 19701 days after 1970-01-01
    let dates = Series::from_date32("d", vec![19701]);
    assert_eq!(dates.len(), 1);
    assert!(dates.is_date());
    assert!(!dates.is_numeric());
    assert!(!dates.is_string());
    assert!(!dates.is_boolean());
    assert_eq!(dates.value_at(0).unwrap(), "2023-12-10");
}

#[test]
fn test_date32_custom_format() {
    let dates = Series::from_date32("d", vec![19701]).with_date_format("DD/MM/YYYY");
    assert_eq!(dates.value_at(0).unwrap(), "10/12/2023");
}

#[test]
fn test_date64_construction_and_display() {
    // 2023-12-10 12:34:56 in ms after epoch
    let ms = 1702211696000i64;
    let dates = Series::from_date64("d", vec![ms]);
    assert_eq!(dates.len(), 1);
    assert!(dates.is_date());
    assert!(dates.value_at(0).unwrap().contains("2023-12-10"));
    assert!(dates.value_at(0).unwrap().contains("12:34:56"));
}

#[test]
fn test_timestamp_ms_construction_and_display() {
    let ms = 1702211696000i64;
    let ts = Series::from_timestamp_ms("ts", vec![ms]);
    assert_eq!(ts.len(), 1);
    assert!(ts.is_date());
    assert!(ts.value_at(0).unwrap().contains("2023-12-10"));
    assert!(ts.value_at(0).unwrap().contains("12:34:56"));
}

#[test]
fn test_date32_filter() {
    let a = Series::from("label", vec!["a", "b", "c"]);
    let d = Series::from_date32("date", vec![19700, 19701, 19702]);
    let mask = Series::from("mask", vec![true, false, true]);
    let df = crate::DataFrame::new(vec![a, d]).unwrap();
    let filtered = df.filter_by_mask(&mask).unwrap();
    assert_eq!(filtered.shape(), (2, 2));
}

#[test]
fn test_date32_sort() {
    let d = Series::from_date32("date", vec![19702, 19700, 19701]);
    let df = crate::DataFrame::new(vec![d]).unwrap();
    let sorted = df.sort_by_asc("date").unwrap();
    let r0 = sorted.get_row(0).unwrap()[0].clone();
    let r2 = sorted.get_row(2).unwrap()[0].clone();
    assert!(r0 < r2, "asc sort: expected {} < {}", r0, r2);
}

#[test]
fn test_date32_min_max() {
    let d = Series::from_date32("date", vec![19702, 19700, 19701]);
    let min = d.min().unwrap().unwrap();
    let max = d.max().unwrap().unwrap();
    assert!(min < max);
}

#[test]
fn test_date32_shuffle() {
    let d = Series::from_date32("date", vec![19700, 19701, 19702, 19703]);
    let df = crate::DataFrame::new(vec![d]).unwrap();
    let shuffled = df.shuffle().unwrap();
    assert_eq!(shuffled.shape(), (4, 1));
}

#[test]
fn test_date32_drop_null() {
    let arr = std::sync::Arc::new(arrow::array::Date32Array::from(vec![
        Some(19700i32),
        None,
        Some(19702),
    ]));
    let d = crate::Series::new("date", arr);
    let cleaned = d.drop_null().unwrap();
    assert_eq!(cleaned.len(), 2);
}

#[test]
fn test_date32_fill_null() {
    let arr = std::sync::Arc::new(arrow::array::Date32Array::from(vec![
        Some(19700i32),
        None,
        Some(19702),
    ]));
    let d = crate::Series::new("date", arr);
    let fill = std::sync::Arc::new(arrow::array::Date32Array::from(vec![
        19799i32, 19800, 19801,
    ]));
    let filler = crate::Series::new("filler", fill);
    let result = d.fill_null(&filler).unwrap();
    assert_eq!(result.len(), 3);
}

// ============================================================================
// DATE EDGE CASES
// ============================================================================

#[test]
fn test_date32_epoch_zero() {
    let d = Series::from_date32("d", vec![0]);
    assert_eq!(d.value_at(0).unwrap(), "1970-01-01");
}

#[test]
fn test_date32_negative_epoch() {
    let d = Series::from_date32("d", vec![-1]);
    assert_eq!(d.value_at(0).unwrap(), "1969-12-31");
}

#[test]
fn test_date32_leap_year_feb29() {
    // 2024 is a leap year. Day 19782 = 2024-02-29
    let d = Series::from_date32("d", vec![19782]);
    assert_eq!(d.value_at(0).unwrap(), "2024-02-29");
}

#[test]
fn test_date32_leap_year_feb28() {
    // Day 19781 = 2024-02-28
    let d = Series::from_date32("d", vec![19781]);
    assert_eq!(d.value_at(0).unwrap(), "2024-02-28");
}

#[test]
fn test_date32_mar1_after_leap_feb29() {
    let d = Series::from_date32("d", vec![19783]);
    assert_eq!(d.value_at(0).unwrap(), "2024-03-01");
}

#[test]
fn test_date_all_null_min() {
    let all_null: Vec<Option<i32>> = vec![None, None, None];
    let arr = std::sync::Arc::new(arrow::array::Date32Array::from(all_null));
    let d = crate::Series::new("date", arr);
    assert!(d.min().unwrap().is_none());
}

#[test]
fn test_date_all_null_max() {
    let all_null: Vec<Option<i32>> = vec![None, None, None];
    let arr = std::sync::Arc::new(arrow::array::Date32Array::from(all_null));
    let d = crate::Series::new("date", arr);
    assert!(d.max().unwrap().is_none());
}

#[test]
fn test_date_all_null_drop_null() {
    let all_null: Vec<Option<i32>> = vec![None, None, None];
    let arr = std::sync::Arc::new(arrow::array::Date32Array::from(all_null));
    let d = crate::Series::new("date", arr);
    assert_eq!(d.drop_null().unwrap().len(), 0);
}

#[test]
fn test_date_sort_with_nulls() {
    // nulls sort first, then ascending by date
    let arr = std::sync::Arc::new(arrow::array::Date32Array::from(vec![
        Some(19702i32),
        None,
        Some(0), // epoch zero = 1970-01-01
    ]));
    let d = crate::Series::new("date", arr);
    let df = crate::DataFrame::new(vec![d]).unwrap();
    let sorted = df.sort_by_asc("date").unwrap();
    assert_eq!(sorted.get_row(0).unwrap()[0], "null");
    assert_eq!(sorted.get_row(1).unwrap()[0], "1970-01-01");
}

// ============================================================================
// DATE COMPARISON OPS
// ============================================================================

#[test]
fn test_date32_comparison_ops() {
    let d = Series::from_date32("d", vec![19700, 19701, 19702]);
    let mask = d.greater_than(crate::Numeric::date32(19700)).unwrap();
    assert_eq!(mask.value_at(0).unwrap(), "false");
    assert_eq!(mask.value_at(1).unwrap(), "true");
    assert_eq!(mask.value_at(2).unwrap(), "true");

    let lte = d.less_than_or_equal(crate::Numeric::date32(19701)).unwrap();
    assert_eq!(lte.value_at(0).unwrap(), "true");
    assert_eq!(lte.value_at(1).unwrap(), "true");
    assert_eq!(lte.value_at(2).unwrap(), "false");

    let eq = d.equal(crate::Numeric::date32(19701)).unwrap();
    assert_eq!(eq.value_at(0).unwrap(), "false");
    assert_eq!(eq.value_at(1).unwrap(), "true");

    let neq = d.not_equal(crate::Numeric::date32(19701)).unwrap();
    assert_eq!(neq.value_at(0).unwrap(), "true");
    assert_eq!(neq.value_at(1).unwrap(), "false");
}

#[test]
fn test_date64_comparison_ops() {
    let ms_per_day: i64 = 86_400_000;
    let d = Series::from_date64("d", vec![19700 * ms_per_day, 19701 * ms_per_day]);
    let mask = d
        .greater_than(crate::Numeric::date64(19700 * ms_per_day))
        .unwrap();
    assert_eq!(mask.value_at(0).unwrap(), "false");
    assert_eq!(mask.value_at(1).unwrap(), "true");
}

#[test]
fn test_timestamp_ms_comparison_ops() {
    let ms: i64 = 1700000000000;
    let ts = Series::from_timestamp_ms("ts", vec![ms, ms + 1000, ms + 2000]);
    let mask = ts
        .less_than(crate::Numeric::timestamp_millis(ms + 1000))
        .unwrap();
    assert_eq!(mask.value_at(0).unwrap(), "true");
    assert_eq!(mask.value_at(1).unwrap(), "false");
    assert_eq!(mask.value_at(2).unwrap(), "false");
}

// ============================================================================
// DATE64 OPERATION TESTS
// ============================================================================

#[test]
fn test_date64_filter() {
    let a = Series::from("label", vec!["x", "y"]);
    let ms_per_day: i64 = 86_400_000;
    let d = Series::from_date64("d", vec![19700 * ms_per_day, 19701 * ms_per_day]);
    let mask = Series::from("mask", vec![true, false]);
    let df = crate::DataFrame::new(vec![a, d]).unwrap();
    let filtered = df.filter_by_mask(&mask).unwrap();
    assert_eq!(filtered.shape(), (1, 2));
}

#[test]
fn test_date64_sort() {
    let ms_per_day: i64 = 86_400_000;
    let d = Series::from_date64(
        "d",
        vec![19702 * ms_per_day, 19700 * ms_per_day, 19701 * ms_per_day],
    );
    let df = crate::DataFrame::new(vec![d]).unwrap();
    let sorted = df.sort_by_asc("d").unwrap();
    let r0 = sorted.get_row(0).unwrap()[0].clone();
    let r2 = sorted.get_row(2).unwrap()[0].clone();
    assert!(r0 < r2, "asc sort: expected {} < {}", r0, r2);
}

#[test]
fn test_date64_shuffle() {
    let ms_per_day: i64 = 86_400_000;
    let d = Series::from_date64(
        "d",
        vec![
            19700 * ms_per_day,
            19701 * ms_per_day,
            19702 * ms_per_day,
            19703 * ms_per_day,
        ],
    );
    let df = crate::DataFrame::new(vec![d]).unwrap();
    let shuffled = df.shuffle().unwrap();
    assert_eq!(shuffled.shape(), (4, 1));
}

#[test]
fn test_date64_drop_null() {
    let ms_per_day: i64 = 86_400_000;
    let arr = std::sync::Arc::new(arrow::array::Date64Array::from(vec![
        Some(19700 * ms_per_day),
        None,
        Some(19702 * ms_per_day),
    ]));
    let d = crate::Series::new("date", arr);
    let cleaned = d.drop_null().unwrap();
    assert_eq!(cleaned.len(), 2);
}

#[test]
fn test_date64_min_max() {
    let ms_per_day: i64 = 86_400_000;
    let d = Series::from_date64(
        "d",
        vec![19702 * ms_per_day, 19700 * ms_per_day, 19701 * ms_per_day],
    );
    let min = d.min().unwrap().unwrap();
    let max = d.max().unwrap().unwrap();
    assert!(min < max);
}

// ============================================================================
// TIMESTAMP OPERATION TESTS
// ============================================================================

#[test]
fn test_timestamp_ms_filter() {
    let ms: i64 = 1700000000000;
    let a = Series::from("label", vec!["x", "y"]);
    let ts = Series::from_timestamp_ms("ts", vec![ms, ms + 1000]);
    let mask = Series::from("mask", vec![true, false]);
    let df = crate::DataFrame::new(vec![a, ts]).unwrap();
    let filtered = df.filter_by_mask(&mask).unwrap();
    assert_eq!(filtered.shape(), (1, 2));
}

#[test]
fn test_timestamp_ms_sort() {
    let ms: i64 = 1700000000000;
    let ts = Series::from_timestamp_ms("ts", vec![ms + 2000, ms, ms + 1000]);
    let df = crate::DataFrame::new(vec![ts]).unwrap();
    let sorted = df.sort_by_asc("ts").unwrap();
    let r0 = sorted.get_row(0).unwrap()[0].clone();
    let r2 = sorted.get_row(2).unwrap()[0].clone();
    assert!(r0 < r2, "asc sort: expected {} < {}", r0, r2);
}

#[test]
fn test_timestamp_ms_shuffle() {
    let ms: i64 = 1700000000000;
    let ts = Series::from_timestamp_ms("ts", vec![ms, ms + 1000, ms + 2000, ms + 3000]);
    let df = crate::DataFrame::new(vec![ts]).unwrap();
    let shuffled = df.shuffle().unwrap();
    assert_eq!(shuffled.shape(), (4, 1));
}

#[test]
fn test_timestamp_ms_drop_null() {
    let ms: i64 = 1700000000000;
    let arr = std::sync::Arc::new(arrow::array::TimestampMillisecondArray::from(vec![
        Some(ms),
        None,
        Some(ms + 2000),
    ]));
    let d = crate::Series::new("ts", arr);
    let cleaned = d.drop_null().unwrap();
    assert_eq!(cleaned.len(), 2);
}

#[test]
fn test_timestamp_ms_min_max() {
    let ms: i64 = 1700000000000;
    let ts = Series::from_timestamp_ms("ts", vec![ms + 2000, ms, ms + 1000]);
    let min = ts.min().unwrap().unwrap();
    let max = ts.max().unwrap().unwrap();
    assert!(min < max);
}

// ============================================================================
// DATE64/TIMESTAMP fill_null TESTS
// ============================================================================

#[test]
fn test_date64_fill_null() {
    let ms_per_day: i64 = 86_400_000;
    let arr = std::sync::Arc::new(arrow::array::Date64Array::from(vec![
        Some(19700 * ms_per_day),
        None,
        Some(19702 * ms_per_day),
    ]));
    let d = crate::Series::new("date", arr);
    let fill = std::sync::Arc::new(arrow::array::Date64Array::from(vec![
        19799 * ms_per_day,
        19800 * ms_per_day,
        19801 * ms_per_day,
    ]));
    let filler = crate::Series::new("filler", fill);
    let result = d.fill_null(&filler).unwrap();
    assert_eq!(result.len(), 3);
    // position 1 was null, should now be the filler's value at position 1
    assert_ne!(result.value_at(1).unwrap(), "null");
    // position 0 was not null, should still be the original value
    assert!(result.value_at(0).unwrap().contains("2023-12-09"));
}

#[test]
fn test_timestamp_ms_fill_null() {
    let ms: i64 = 1700000000000;
    let arr = std::sync::Arc::new(arrow::array::TimestampMillisecondArray::from(vec![
        Some(ms),
        None,
        Some(ms + 2000),
    ]));
    let d = crate::Series::new("ts", arr);
    let fill = std::sync::Arc::new(arrow::array::TimestampMillisecondArray::from(vec![
        ms + 9999,
        ms + 9999,
        ms + 9999,
    ]));
    let filler = crate::Series::new("filler", fill);
    let result = d.fill_null(&filler).unwrap();
    assert_eq!(result.len(), 3);
    // The filled position (index 1) should have the filler's value
    assert!(result.value_at(1).unwrap() != "null");
}

// ============================================================================
// YEAR BOUNDARY EDGE CASES
// ============================================================================

#[test]
fn test_date32_year_boundary_dec31_to_jan1() {
    // 2023-12-31 (day 19722) to 2024-01-01 (day 19723)
    let d = Series::from_date32("d", vec![19722, 19723]);
    assert_eq!(d.value_at(0).unwrap(), "2023-12-31");
    assert_eq!(d.value_at(1).unwrap(), "2024-01-01");
}

#[test]
fn test_date32_sort_cross_year_boundary() {
    let d = Series::from_date32("d", vec![19723, 19722]); // Jan 1, Dec 31
    let df = crate::DataFrame::new(vec![d]).unwrap();
    let sorted = df.sort_by_asc("d").unwrap();
    assert_eq!(sorted.get_row(0).unwrap()[0], "2023-12-31");
    assert_eq!(sorted.get_row(1).unwrap()[0], "2024-01-01");
}

#[test]
fn test_date32_far_future() {
    // Day 110000 = year ~2271
    let d = Series::from_date32("d", vec![110000]);
    let val = d.value_at(0).unwrap();
    assert!(val.starts_with("2")); // year in 2200s
    assert_ne!(val, "null");
    assert!(!val.contains("Unsupported"));
}

#[test]
fn test_date32_groupby_max_integration() {
    let cat = Series::from("cat", vec!["a", "b", "a"]);
    let dates = Series::from_date32("d", vec![19701, 19700, 19702]);
    let df = crate::DataFrame::new(vec![cat, dates]).unwrap();
    let grouped = df.group_by("cat").unwrap();
    let max_result = grouped.max("d").unwrap();
    assert_eq!(max_result.shape().0, 2);
}

// ============================================================================
// try_into_date32 TESTS
// ============================================================================

#[test]
fn test_try_into_date32_iso_format() {
    let strings = Series::from("dates", vec!["2023-12-10", "2024-01-15"]);
    let dates = strings.try_into_date32("YYYY-MM-DD").unwrap();
    assert!(dates.is_date());
    assert_eq!(dates.len(), 2);
    assert_eq!(dates.value_at(0).unwrap(), "2023-12-10");
    assert_eq!(dates.value_at(1).unwrap(), "2024-01-15");
}

#[test]
fn test_try_into_date32_dd_mm_yyyy_format() {
    let strings = Series::from("dates", vec!["10/12/2023", "15/01/2024"]);
    let dates = strings.try_into_date32("DD/MM/YYYY").unwrap();
    assert_eq!(dates.len(), 2);
    assert_eq!(dates.value_at(0).unwrap(), "2023-12-10");
    assert_eq!(dates.value_at(1).unwrap(), "2024-01-15");
}

#[test]
fn test_try_into_date32_with_nulls() {
    let strings = Series::from("dates", vec![Some("2023-12-10"), None, Some("2024-01-15")]);
    let dates = strings.try_into_date32("YYYY-MM-DD").unwrap();
    assert_eq!(dates.len(), 3);
    assert_eq!(dates.value_at(1).unwrap(), "null");
}

#[test]
fn test_try_into_date32_invalid_format() {
    let strings = Series::from("dates", vec!["not-a-date"]);
    let result = strings.try_into_date32("YYYY-MM-DD");
    assert!(result.is_err());
}

#[test]
fn test_try_into_date32_non_string_series() {
    let ints = Series::from("ints", vec![1, 2, 3]);
    let result = ints.try_into_date32("YYYY-MM-DD");
    assert!(result.is_err());
}

// ============================================================================
// PRE-1970 DATE REGRESSION TESTS (rem_euclid bug fix)
// ============================================================================

#[test]
fn test_date64_pre_1970_not_midnight() {
    // 1 second before epoch midnight: 1969-12-31 23:59:59
    let ms: i64 = -1000;
    let d = Series::from_date64("d", vec![ms]);
    let val = d.value_at(0).unwrap();
    assert!(val.contains("1969-12-31"));
    assert!(val.contains("23:59:59"));
}

#[test]
fn test_timestamp_ms_pre_1970_not_midnight() {
    let ms: i64 = -1000;
    let ts = Series::from_timestamp_ms("ts", vec![ms]);
    let val = ts.value_at(0).unwrap();
    assert!(val.contains("1969-12-31"));
    assert!(val.contains("23:59:59"));
}

#[test]
fn test_date64_pre_1970_midnight() {
    // exact midnight: 1969-12-31 00:00:00
    let ms: i64 = -86400000;
    let d = Series::from_date64("d", vec![ms]);
    let val = d.value_at(0).unwrap();
    assert!(val.contains("1969-12-31"));
    assert!(val.contains("00:00:00"));
}

#[test]
fn test_groupby_minmax_all_null_returns_nan() {
    let cat = Series::from("cat", vec!["x", "x"]);
    let vals: Vec<Option<i32>> = vec![None, None];
    let d = Series::from("v", vals);
    let df = crate::DataFrame::new(vec![cat, d]).unwrap();
    let grouped = df.group_by("cat").unwrap();
    let min_r = grouped.min("v").unwrap();
    let max_r = grouped.max("v").unwrap();
    // Single group with all nulls should return NaN
    let min_val: f64 = min_r
        .select("v_min")
        .unwrap()
        .value_at(0)
        .unwrap()
        .parse()
        .unwrap();
    let max_val: f64 = max_r
        .select("v_max")
        .unwrap()
        .value_at(0)
        .unwrap()
        .parse()
        .unwrap();
    assert!(min_val.is_nan());
    assert!(max_val.is_nan());
}
