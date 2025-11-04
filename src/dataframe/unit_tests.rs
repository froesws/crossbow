use crate::{CrossbowError, DataFrame, Series};

// Helper function for creating a simple DF
fn create_test_dataframe() -> DataFrame {
    let s1 = Series::from("col A", vec![1, 2, 3]); // Infers i32
    let s2 = Series::from("col B", vec!["a", "b", "c"]);
    DataFrame::new(vec![s1, s2]).unwrap()
}

#[test]
fn test_new_dataframe_success() {
    let s1 = Series::from("col A", vec![1i64, 2, 3]); // Use i64
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
    let s2 = Series::from("col B", vec![4, 5]); // Mismatched len
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
    let s2 = Series::from("col A", vec![4, 5, 6]); // Duplicate name
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

// --- New Tests for Filter & CSV ---

#[test]
fn test_filter_dataframe() {
    let s1 = Series::from("ages", vec![10i64, 20, 30, 40]);
    let s2 = Series::from("names", vec!["a", "b", "c", "d"]);
    let df = DataFrame::new(vec![s1, s2]).unwrap();

    // Create mask: ages > 25
    let mask = df.select("ages").unwrap().gt_i64(25).unwrap();

    let filtered_df = df.filter(&mask).unwrap();

    // Check shape and names
    assert_eq!(filtered_df.shape(), (2, 2));
    assert_eq!(filtered_df.get_column_names(), vec!["ages", "names"]);

    // Verify content using the crate-visible accessors
    let filtered_ages = filtered_df.select("ages").unwrap();
    assert_eq!(filtered_ages.as_i64_slice().unwrap(), &[Some(30), Some(40)]);

    let filtered_names = filtered_df.select("names").unwrap();
    let expected_names = &[Some("c".to_string()), Some("d".to_string())];
    assert_eq!(filtered_names.as_string_slice().unwrap(), expected_names);
}

#[test]
fn test_filter_with_nulls_in_mask() {
    let s1 = Series::from("ages", vec![10i64, 20, 30, 40]);
    // Mask: [true, false, null, true]
    let mask = Series::from("mask", vec![Some(true), Some(false), None, Some(true)]);

    let df = DataFrame::new(vec![s1]).unwrap();
    let filtered_df = df.filter(&mask).unwrap();

    // Nulls in mask should be filtered out
    assert_eq!(filtered_df.shape(), (2, 1));
    let filtered_ages = filtered_df.select("ages").unwrap();
    assert_eq!(filtered_ages.as_i64_slice().unwrap(), &[Some(10), Some(40)]);
}

// We need these imports for the I/O test
use crate::types::DataType;
use std::fs::File;
use std::io::Write;

#[test]
fn test_read_csv() {
    let csv_content = "id,name,is_user,score\n1,Alice,true,9.5\n2,Bob,false,\n3,Charlie,true,8.1";
    let file_path = "temp_test_data.csv";

    // Create the dummy file
    let mut file = File::create(file_path).unwrap();
    writeln!(file, "{}", csv_content).unwrap();
    drop(file); // Close the file

    // Run the test
    let df_result = DataFrame::read_csv(file_path);
    assert!(df_result.is_ok());
    let df = df_result.unwrap();

    // Check metadata
    assert_eq!(df.shape(), (3, 4));
    assert_eq!(
        df.get_column_names(),
        vec!["id", "name", "is_user", "score"]
    );

    // Check inferred dtypes
    assert_eq!(df.select("id").unwrap().dtype(), DataType::Int64);
    assert_eq!(df.select("name").unwrap().dtype(), DataType::String);
    assert_eq!(df.select("is_user").unwrap().dtype(), DataType::Boolean);
    assert_eq!(df.select("score").unwrap().dtype(), DataType::Float64);

    // Check content
    let id_col = df.select("id").unwrap();
    assert_eq!(
        id_col.as_i64_slice().unwrap(),
        &[Some(1_i64), Some(2), Some(3)]
    );

    let name_col = df.select("name").unwrap();
    let expected_names = &[
        Some("Alice".to_string()),
        Some("Bob".to_string()),
        Some("Charlie".to_string()),
    ];
    assert_eq!(name_col.as_string_slice().unwrap(), expected_names);

    let user_col = df.select("is_user").unwrap();
    assert_eq!(
        user_col.as_bool_slice().unwrap(),
        &[Some(true), Some(false), Some(true)]
    );

    let score_col = df.select("score").unwrap();
    assert_eq!(
        score_col.as_f64_slice().unwrap(),
        &[Some(9.5), None, Some(8.1)]
    );

    // Cleanup
    std::fs::remove_file(file_path).unwrap();
}

#[test]
fn test_read_csv_file_not_found() {
    let result = DataFrame::read_csv("a_file_that_does_not_exist.csv");
    assert!(result.is_err());
    // Check that it's an IO error (which comes from csv::Reader::from_path)
    assert!(matches!(result.err().unwrap(), CrossbowError::IoError(_)));
}
