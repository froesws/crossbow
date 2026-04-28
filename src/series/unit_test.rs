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
    assert!(series_i32.as_primitive::<arrow::array::Float64Array>().is_none());
    
    let series_f64 = Series::from("floats", vec![1.1, 2.2, 3.3]);
    assert!(series_f64.as_primitive::<arrow::array::Int32Array>().is_none());
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
    assert_eq!(series.get_value_as_string(0).parse::<i32>().unwrap(), i32::MAX);
    assert_eq!(series.get_value_as_string(1).parse::<i32>().unwrap(), i32::MIN);
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
    assert!(series.get_value_as_string(0).contains("inf") || series.get_value_as_string(0).contains("∞"));
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
    assert!(std_val.is_nan(), "Expected NaN for single value std, got {}", std_val);
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
    assert!(var_val.is_nan(), "Expected NaN for all-null var, got {}", var_val);
}

#[test]
fn test_var_single_value_returns_nan() {
    let single = Series::from("single", vec![42i32]);
    let var_val = single.var().unwrap();
    assert!(var_val.is_nan(), "Expected NaN for single value var, got {}", var_val);
}

