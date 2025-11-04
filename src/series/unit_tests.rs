use crate::{CrossbowError, Series, types::DataType};

#[test]
fn test_series_from_i32() {
    let series = Series::from("my_integers", vec![10, 20, 30]);
    assert_eq!(series.name(), "my_integers");
    assert_eq!(series.len(), 3);
    assert_eq!(series.dtype(), DataType::Int32); // Updated

    // Check values using new accessor
    let slice = series.as_i32_slice().unwrap();
    assert_eq!(slice, &[Some(10), Some(20), Some(30)]);
}

#[test]
fn test_series_from_optional_i32() {
    let series = Series::from("my_optional_integers", vec![Some(10), None, Some(30)]);
    assert_eq!(series.name(), "my_optional_integers");
    assert_eq!(series.len(), 3);
    assert_eq!(series.dtype(), DataType::Int32); // Updated
    assert!(series.is_null(1)); // Updated from data().is_null()
    assert!(!series.is_null(0));

    // Check values
    let slice = series.as_i32_slice().unwrap();
    assert_eq!(slice, &[Some(10), None, Some(30)]);
}

#[test]
fn test_series_from_f64() {
    let series = Series::from("my_floats", vec![1.1, 2.2, 3.3]);
    assert_eq!(series.name(), "my_floats");
    assert_eq!(series.len(), 3);
    assert_eq!(series.dtype(), DataType::Float64); // Updated
    let slice = series.as_f64_slice().unwrap();
    assert_eq!(slice, &[Some(1.1), Some(2.2), Some(3.3)]);
}

#[test]
fn test_series_from_bool() {
    let series = Series::from("my_booleans", vec![true, false, true]);
    assert_eq!(series.name(), "my_booleans");
    assert_eq!(series.len(), 3);
    assert_eq!(series.dtype(), DataType::Boolean); // Updated
    let slice = series.as_bool_slice().unwrap();
    assert_eq!(slice, &[Some(true), Some(false), Some(true)]);
}

#[test]
fn test_series_from_str() {
    let series = Series::from("fruits", vec!["apple", "banana", "cherry"]);
    assert_eq!(series.name(), "fruits");
    assert_eq!(series.len(), 3);
    assert_eq!(series.dtype(), DataType::String); // Updated from Utf8
    let slice = series.as_string_slice().unwrap();
    assert_eq!(
        slice,
        &[
            Some("apple".to_string()),
            Some("banana".to_string()),
            Some("cherry".to_string())
        ]
    );
}

#[test]
fn test_series_from_optional_str() {
    let series = Series::from("optional_fruits", vec![Some("apple"), None, Some("cherry")]);
    assert_eq!(series.name(), "optional_fruits");
    assert_eq!(series.len(), 3);
    assert_eq!(series.dtype(), DataType::String); // Updated from Utf8
    assert!(series.is_null(1)); // Updated
    let slice = series.as_string_slice().unwrap();
    assert_eq!(
        slice,
        &[Some("apple".to_string()), None, Some("cherry".to_string())]
    );
}

#[test]
fn test_series_from_string() {
    let data = vec!["apple".to_string(), "banana".to_string()];
    let series = Series::from("string_fruits", data);
    assert_eq!(series.name(), "string_fruits");
    assert_eq!(series.len(), 2);
    assert_eq!(series.dtype(), DataType::String); // Updated from Utf8
    let slice = series.as_string_slice().unwrap();
    assert_eq!(
        slice,
        &[Some("apple".to_string()), Some("banana".to_string())]
    );
}

// Removed test_as_primitive_downcast (was Arrow-specific)
// Removed test_get_value_as_string (function was removed)
// Removed test_get_value_as_string_out_of_bounds (function was removed)

#[test]
fn test_series_len() {
    let series = Series::from("length_test", vec![1, 2, 3, 4, 5]);
    assert_eq!(series.len(), 5);
}

// --- New Tests for Filter/Ops ---

#[test]
fn test_series_gt_i64() {
    let series = Series::from("numbers", vec![Some(10i64), Some(20), Some(30), None]);
    let mask = series.gt_i64(15).unwrap();

    assert_eq!(mask.len(), 4);
    assert_eq!(mask.dtype(), DataType::Boolean);

    // Check values using new accessor
    let slice = mask.as_bool_slice().unwrap();
    assert_eq!(slice, &[Some(false), Some(true), Some(true), None]);
}

#[test]
fn test_series_filter() {
    let series = Series::from("names", vec![Some("a"), Some("b"), Some("c"), Some("d")]);
    // Mask: [true, false, true, null]
    let mask = Series::from("mask", vec![Some(true), Some(false), Some(true), None]);

    let filtered = series.filter(&mask).unwrap();

    assert_eq!(filtered.len(), 2);
    assert_eq!(filtered.dtype(), DataType::String);

    // Check values
    let slice = filtered.as_string_slice().unwrap();
    assert_eq!(slice, &[Some("a".to_string()), Some("c".to_string())]);
}

#[test]
fn test_series_filter_mismatched_length() {
    let series = Series::from("names", vec!["a", "b", "c"]);
    let mask = Series::from("mask", vec![true, false]); // Wrong length

    let result = series.filter(&mask);
    assert!(result.is_err());
    assert!(matches!(
        result.err().unwrap(),
        CrossbowError::MismatchedColumnLengths
    ));
}

#[test]
fn test_series_filter_wrong_mask_type() {
    let series = Series::from("names", vec!["a", "b", "c"]);
    let mask = Series::from("mask", vec![1, 2, 3]); // Wrong type

    let result = series.filter(&mask);
    assert!(result.is_err());
    assert!(matches!(
        result.err().unwrap(),
        CrossbowError::OperationNotSupported(_)
    ));
}
