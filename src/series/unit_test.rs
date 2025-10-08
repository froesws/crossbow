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

    // Downcast bem-sucedido
    if let Some(int_array) = series_i32.as_primitive::<Int32Array>() {
        assert_eq!(int_array.value(0), 1);
    } else {
        panic!("Downcast to Int32Array should succeed");
    }

    // Downcast mal-sucedido
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
