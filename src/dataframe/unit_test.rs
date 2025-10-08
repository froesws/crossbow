use crate::{CrossbowError, DataFrame, Series};

fn create_test_dataframe() -> DataFrame {
    let s1 = Series::from("col A", vec![1, 2, 3]);
    let s2 = Series::from("col B", vec!["a", "b", "c"]);
    DataFrame::new(vec![s1, s2]).unwrap()
}

#[test]
fn test_new_dataframe_success() {
    let s1 = Series::from("col A", vec![1i64, 2, 3]);
    let s2 = Series::from("col B", vec![4.0f64, 5.0, 6.0]);
    let df = DataFrame::new(vec![s1, s2]);
    assert!(df.is_ok());
    let df = df.unwrap();
    assert_eq!(df.shape(), (3, 2));
}

#[test]
fn test_new_dataframe_empty() {
    let df = DataFrame::new(vec![]);
    assert!(df.is_ok());
    let df = df.unwrap();
    assert_eq!(df.shape(), (0, 0));
}

#[test]
fn test_new_dataframe_mismatched_lengths() {
    let s1 = Series::from("col A", vec![1, 2, 3]);
    let s2 = Series::from("col B", vec![4, 5]);
    let df = DataFrame::new(vec![s1, s2]);
    assert!(df.is_err());
    assert!(matches!(
        df.err().unwrap(),
        CrossbowError::MismatchedColumnLengths
    ));
}

#[test]
fn test_new_dataframe_duplicate_names() {
    let s1 = Series::from("col A", vec![1, 2, 3]);
    let s2 = Series::from("col A", vec![4, 5, 6]);
    let df = DataFrame::new(vec![s1, s2]);
    assert!(df.is_err());
    assert!(matches!(
        df.err().unwrap(),
        CrossbowError::DuplicateColumnName(name) if name == "col A"
    ));
}

#[test]
fn test_shape() {
    let df = create_test_dataframe();
    assert_eq!(df.shape(), (3, 2));
}

#[test]
fn test_shape_empty_dataframe() {
    let df = DataFrame::new(vec![]).unwrap();
    assert_eq!(df.shape(), (0, 0));
}

#[test]
fn test_shape_zero_rows() {
    let s1: Series = Series::from("col A", Vec::<i32>::new());
    let s2: Series = Series::from("col B", Vec::<&str>::new());
    let df = DataFrame::new(vec![s1, s2]).unwrap();
    assert_eq!(df.shape(), (0, 2));
}

#[test]
fn test_get_column_names() {
    let df = create_test_dataframe();
    let names = df.get_column_names();
    assert_eq!(names, vec!["col A", "col B"]);
}

#[test]
fn test_get_column_names_empty_dataframe() {
    let df = DataFrame::new(vec![]).unwrap();
    let names = df.get_column_names();
    assert!(names.is_empty());
}

#[test]
fn test_select_column_success() {
    let s1 = Series::from("col A", vec![1, 2, 3]);
    let s2 = Series::from("col B", vec!["a", "b", "c"]);
    let df = DataFrame::new(vec![s1.clone(), s2]).unwrap();

    let selected_series = df.select("col A");
    assert!(selected_series.is_ok());

    let selected_series = selected_series.unwrap();
    assert_eq!(selected_series.name(), s1.name());
    assert_eq!(selected_series.len(), s1.len());
}

#[test]
fn test_select_column_not_found() {
    let df = create_test_dataframe();
    let result = df.select("col C");
    assert!(result.is_err());
    assert!(matches!(
        result.err().unwrap(),
        CrossbowError::ColumnNotFound(name) if name == "col C"
    ));
}
