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

// ============================================================================
// NEW TESTS: EDGE CASES & HIGH PRIORITY COVERAGE
// ============================================================================

// --- SINGLE ROW DATAFRAME OPERATIONS ---

#[test]
fn test_single_row_dataframe_operations() {
    let s1 = Series::from("col A", vec![42]);
    let s2 = Series::from("col B", vec!["x"]);
    let df = DataFrame::new(vec![s1, s2]).unwrap();
    
    assert_eq!(df.shape(), (1, 2));
    assert_eq!(df.get_column_names(), vec!["col A", "col B"]);
    
    let row = df.get_row(0).unwrap();
    assert_eq!(row.len(), 2);
}

#[test]
fn test_single_column_dataframe() {
    let s = Series::from("only_col", vec![1, 2, 3, 4]);
    let df = DataFrame::new(vec![s]).unwrap();
    
    assert_eq!(df.shape(), (4, 1));
    assert_eq!(df.get_column_names(), vec!["only_col"]);
}

#[test]
fn test_single_row_filter() {
    let s1 = Series::from("val", vec![100]);
    let df = DataFrame::new(vec![s1]).unwrap();
    
    let mask = Series::from("mask", vec![true]);
    let filtered = df.filter_by_mask(&mask).unwrap();
    assert_eq!(filtered.shape(), (1, 1));
    
    let mask_false = Series::from("mask", vec![false]);
    let empty = df.filter_by_mask(&mask_false).unwrap();
    assert_eq!(empty.shape(), (0, 1));
}

#[test]
fn test_single_row_sort() {
    let s1 = Series::from("val", vec![42]);
    let df = DataFrame::new(vec![s1]).unwrap();
    
    let sorted_asc = df.sort_by("val", true).unwrap();
    assert_eq!(sorted_asc.shape(), (1, 1));
    assert_eq!(sorted_asc.get_row(0).unwrap()[0], "42");
}

// --- FILTER EDGE CASES ---

#[test]
fn test_filter_all_false_mask() {
    let s1 = Series::from("col A", vec![1, 2, 3, 4, 5]);
    let s2 = Series::from("col B", vec!["a", "b", "c", "d", "e"]);
    let df = DataFrame::new(vec![s1, s2]).unwrap();
    
    let all_false = Series::from("mask", vec![false, false, false, false, false]);
    let result = df.filter_by_mask(&all_false).unwrap();
    assert_eq!(result.shape(), (0, 2));
    assert_eq!(result.get_column_names(), vec!["col A", "col B"]);
}

#[test]
fn test_filter_all_true_mask() {
    let s1 = Series::from("col A", vec![1, 2, 3, 4, 5]);
    let s2 = Series::from("col B", vec!["a", "b", "c", "d", "e"]);
    let df = DataFrame::new(vec![s1.clone(), s2.clone()]).unwrap();
    
    let all_true = Series::from("mask", vec![true, true, true, true, true]);
    let result = df.filter_by_mask(&all_true).unwrap();
    assert_eq!(result.shape(), df.shape());
    
    // Verify data is preserved
    for i in 0..5 {
        let orig_row = df.get_row(i).unwrap();
        let filt_row = result.get_row(i).unwrap();
        assert_eq!(orig_row, filt_row);
    }
}

#[test]
fn test_filter_mask_wrong_length() {
    let s1 = Series::from("col A", vec![1, 2, 3]);
    let df = DataFrame::new(vec![s1]).unwrap();
    
    let wrong_len_mask = Series::from("mask", vec![true, false]);
    let result = df.filter_by_mask(&wrong_len_mask);
    assert!(result.is_err());
}

#[test]
fn test_filter_result_to_new_dataframe() {
    let s1 = Series::from("val", vec![1, 2, 3, 4, 5]);
    let df = DataFrame::new(vec![s1]).unwrap();
    
    let mask = Series::from("mask", vec![true, false, true, false, true]);
    let filtered = df.filter_by_mask(&mask).unwrap();
    assert_eq!(filtered.shape(), (3, 1));
    
    // Verify the filtered result can be used as a new DataFrame
    let re_filtered = filtered.filter_by_mask(&Series::from("mask2", vec![true, true, true])).unwrap();
    assert_eq!(re_filtered.shape(), (3, 1));
}

// --- SORT EDGE CASES ---

#[test]
fn test_sort_preserves_row_integrity() {
    let s1 = Series::from("age", vec![30, 25, 35, 25, 40]);
    let s2 = Series::from("name", vec!["Alice", "Bob", "Charlie", "David", "Eve"]);
    let df = DataFrame::new(vec![s1, s2]).unwrap();
    
    let sorted = df.sort_by("age", true).unwrap();
    
    // Verify that names follow their corresponding ages
    let ages_col = sorted.select("age").unwrap();
    let names_col = sorted.select("name").unwrap();
    
    // Check names are with their correct ages after sort
    assert_eq!(ages_col.get_value_as_string(0), "25");
    assert_eq!(ages_col.get_value_as_string(1), "25");
    
    // Names should be Bob and David in some order (both age 25)
    let name_0 = names_col.get_value_as_string(0);
    let name_1 = names_col.get_value_as_string(1);
    assert!((name_0 == "\"Bob\"" && name_1 == "\"David\"") || 
            (name_0 == "\"David\"" && name_1 == "\"Bob\""));
}

#[test]
fn test_sort_uniform_values() {
    let s1 = Series::from("val", vec![5, 5, 5, 5, 5]);
    let s2 = Series::from("idx", vec![0, 1, 2, 3, 4]);
    let df = DataFrame::new(vec![s1, s2]).unwrap();
    
    let sorted = df.sort_by("val", true).unwrap();
    assert_eq!(sorted.shape(), (5, 2));
    
    // All values should still be 5
    let vals = sorted.select("val").unwrap();
    for i in 0..5 {
        assert_eq!(vals.get_value_as_string(i), "5");
    }
}

#[test]
fn test_sort_descending_strings() {
    let s1 = Series::from("names", vec!["Charlie", "Alice", "Bob", "David"]);
    let df = DataFrame::new(vec![s1]).unwrap();
    
    let sorted = df.sort_by("names", false).unwrap();
    let names = sorted.select("names").unwrap();
    
    // Should be in reverse alphabetical order
    let v0 = names.get_value_as_string(0);
    let v1 = names.get_value_as_string(1);
    let v2 = names.get_value_as_string(2);
    let v3 = names.get_value_as_string(3);
    
    assert!(v0 >= v1);
    assert!(v1 >= v2);
    assert!(v2 >= v3);
}

// --- COLUMN MANAGEMENT EDGE CASES ---

#[test]
fn test_add_column_wrong_length() {
    let s1 = Series::from("col A", vec![1, 2, 3]);
    let df = DataFrame::new(vec![s1]).unwrap();
    
    let wrong_len = Series::from("col B", vec![1, 2]);
    let result = df.add_column(wrong_len);
    assert!(result.is_err());
}

#[test]
fn test_add_column_duplicate_name() {
    let s1 = Series::from("col A", vec![1, 2, 3]);
    let df = DataFrame::new(vec![s1]).unwrap();
    
    let duplicate = Series::from("col A", vec![4, 5, 6]);
    let result = df.add_column(duplicate);
    assert!(result.is_err());
}

#[test]
fn test_rename_to_existing_name() {
    let s1 = Series::from("col A", vec![1, 2, 3]);
    let s2 = Series::from("col B", vec![4, 5, 6]);
    let df = DataFrame::new(vec![s1, s2]).unwrap();
    
    let result = df.rename_column("col A", "col B");
    assert!(result.is_err());
}

#[test]
fn test_rename_nonexistent_column() {
    let s1 = Series::from("col A", vec![1, 2, 3]);
    let df = DataFrame::new(vec![s1]).unwrap();
    
    let result = df.rename_column("col X", "col Y");
    assert!(result.is_err());
}

#[test]
fn test_remove_nonexistent_column() {
    let s1 = Series::from("col A", vec![1, 2, 3]);
    let df = DataFrame::new(vec![s1]).unwrap();
    
    let result = df.remove_column("col B");
    assert!(result.is_err());
}

#[test]
fn test_column_operations_sequence() {
    let s1 = Series::from("col A", vec![1, 2, 3]);
    let df = DataFrame::new(vec![s1]).unwrap();
    
    // Add column
    let bonus = Series::from("bonus", vec![10, 20, 30]);
    let df = df.add_column(bonus).unwrap();
    assert_eq!(df.shape(), (3, 2));
    
    // Rename column
    let df = df.rename_column("bonus", "bonus_new").unwrap();
    assert!(df.select("bonus_new").is_ok());
    
    // Remove column
    let df = df.remove_column("bonus_new").unwrap();
    assert_eq!(df.shape(), (3, 1));
}

// --- GET ROW/ROWS EDGE CASES ---

#[test]
fn test_get_row_last_row() {
    let s1 = Series::from("val", vec![1, 2, 3, 4, 5]);
    let df = DataFrame::new(vec![s1]).unwrap();
    
    let last_row = df.get_row(4).unwrap();
    assert_eq!(last_row[0], "5");
    
    let out_of_bounds = df.get_row(5);
    assert!(out_of_bounds.is_err());
}

#[test]
fn test_get_rows_empty_indices() {
    let s1 = Series::from("val", vec![1, 2, 3]);
    let df = DataFrame::new(vec![s1]).unwrap();
    
    let rows = df.get_rows(&[]).unwrap();
    assert_eq!(rows.len(), 0);
}

#[test]
fn test_get_rows_duplicates() {
    let s1 = Series::from("val", vec![1, 2, 3]);
    let df = DataFrame::new(vec![s1]).unwrap();
    
    let rows = df.get_rows(&[0, 0, 1, 1]).unwrap();
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[0][0], "1");
    assert_eq!(rows[1][0], "1");
    assert_eq!(rows[2][0], "2");
    assert_eq!(rows[3][0], "2");
}

#[test]
fn test_get_rows_out_of_order() {
    let s1 = Series::from("val", vec![1, 2, 3, 4, 5]);
    let df = DataFrame::new(vec![s1]).unwrap();
    
    let rows = df.get_rows(&[4, 2, 0]).unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0][0], "5");
    assert_eq!(rows[1][0], "3");
    assert_eq!(rows[2][0], "1");
}

// --- GROUP BY EDGE CASES ---

#[test]
fn test_groupby_single_group() {
    let s1 = Series::from("category", vec!["A", "A", "A", "A"]);
    let s2 = Series::from("value", vec![10, 20, 30, 40]);
    let df = DataFrame::new(vec![s1, s2]).unwrap();
    
    let grouped = df.group_by("category").unwrap();
    let result = grouped.count().unwrap();
    
    assert_eq!(result.shape().0, 1);
    assert_eq!(result.get_row(0).unwrap()[1], "4");
}

#[test]
fn test_groupby_all_unique() {
    let s1 = Series::from("id", vec![1, 2, 3, 4, 5]);
    let s2 = Series::from("val", vec![10, 20, 30, 40, 50]);
    let df = DataFrame::new(vec![s1, s2]).unwrap();
    
    let grouped = df.group_by("id").unwrap();
    let result = grouped.count().unwrap();
    
    assert_eq!(result.shape().0, 5);
    for i in 0..5 {
        assert_eq!(result.get_row(i).unwrap()[1], "1");
    }
}

fn load_departments() -> DataFrame {
    let dept_id = Series::from("department_id", vec![10i32, 20, 30, 50]);
    let dept_name = Series::from("department_name", vec!["Engineering", "Marketing", "Sales", "Finance"]);
    DataFrame::new(vec![dept_id, dept_name]).unwrap()
}

fn load_employees() -> DataFrame {
    let emp_id = Series::from("employee_id", vec![1i32, 2, 3, 4, 5]);
    let emp_name = Series::from("employee_name", vec!["Alice", "Bob", "Charlie", "David", "Eve"]);
    let dept_id = Series::from("department_id", vec![10i32, 20, 40, 10, 30]);
    DataFrame::new(vec![emp_id, emp_name, dept_id]).unwrap()
}

fn make_int_df(columns: Vec<(&str, Vec<i32>)>) -> DataFrame {
    let series: Vec<Series> = columns.into_iter().map(|(name, vals)| {
        Series::from(&name.to_string(), vals)
    }).collect();
    DataFrame::new(series).unwrap()
}

#[test]
fn test_join_inner_basic() {
    let emp = load_employees();
    let dept = load_departments();
    let result = emp.join_inner(&dept, "department_id", "department_id").unwrap();

    assert_eq!(result.shape().0, 4);
    let cols = result.get_column_names();
    assert!(cols.contains(&"employee_name"));
    assert!(cols.contains(&"department_name"));
    assert!(!cols.contains(&"employee_id_right"));

    let mut names: Vec<_> = (0..result.shape().0).map(|i| {
        let r = result.get_row(i).unwrap();
        format!("{}|{}", r[1], r[3])
    }).collect();
    names.sort();
    assert_eq!(names, vec![
        "\"Alice\"|\"Engineering\"",
        "\"Bob\"|\"Marketing\"",
        "\"David\"|\"Engineering\"",
        "\"Eve\"|\"Sales\"",
    ]);
}

#[test]
fn test_join_inner_no_matches_returns_empty() {
    let emp = load_employees();
    let small = make_int_df(vec![("id", vec![99i32]), ("x", vec![0])]);
    let result = emp.join_inner(&small, "employee_id", "id").unwrap();
    assert_eq!(result.shape().0, 0);
}

#[test]
fn test_join_inner_one_to_many() {
    let left = make_int_df(vec![("a", vec![1i32, 2]), ("b", vec![10, 20])]);
    let right = DataFrame::new(vec![
        Series::from("a", vec![1i32, 1, 2]),
        Series::from("c", vec!["foo", "bar", "baz"]),
    ]).unwrap();
    let result = left.join_inner(&right, "a", "a").unwrap();
    assert_eq!(result.shape().0, 3);
}

#[test]
fn test_join_left_keeps_unmatched() {
    let emp = load_employees();
    let dept = load_departments();
    let result = emp.join_left(&dept, "department_id", "department_id").unwrap();

    assert_eq!(result.shape().0, 5);
    let charlie = result.get_rows(&[2]).unwrap();
    assert_eq!(charlie[0][1], "\"Charlie\"");
    assert_eq!(charlie[0][3], "null");
}

#[test]
fn test_join_left_all_right_cols_null_for_unmatched() {
    let emp = load_employees();
    let dept = load_departments();
    let result = emp.join_left(&dept, "department_id", "department_id").unwrap();

    let row3 = result.get_row(2).unwrap();
    assert_eq!(row3[1], "\"Charlie\"");
    assert!(row3[3] == "null");
}

#[test]
fn test_join_outer_keeps_all_both_sides() {
    let emp = load_employees();
    let dept = load_departments();
    let result = emp.join_outer(&dept, "department_id", "department_id").unwrap();

    let rows = result.shape().0;
    assert!(rows >= 5);
    assert!(rows <= 7);

    let seen_depts: Vec<String> = (0..rows)
        .map(|i| {
            let r = result.get_row(i).unwrap();
            format!("{}|{}", r[3], r[1])
        })
        .collect();

    assert!(seen_depts.contains(&"\"Finance\"|null".to_string()));
    assert!(seen_depts.contains(&"null|\"Charlie\"".to_string()));
}

#[test]
fn test_join_outer_unmatched_right_row() {
    let emp = load_employees();
    let dept = load_departments();
    let result = emp.join_outer(&dept, "department_id", "department_id").unwrap();

    let finance_row = (0..result.shape().0)
        .find(|&i| {
            let r = result.get_row(i).unwrap();
            r[3] == "\"Finance\""
        })
        .map(|i| result.get_row(i).unwrap());

    assert!(finance_row.is_some());
    let fr = finance_row.unwrap();
    assert_eq!(fr[3], "\"Finance\"");
    assert_eq!(fr[1], "null");
}

#[test]
fn test_join_outer_unmatched_left_row() {
    let emp = load_employees();
    let dept = load_departments();
    let result = emp.join_outer(&dept, "department_id", "department_id").unwrap();

    let charlie_row = (0..result.shape().0)
        .find(|&i| {
            let r = result.get_row(i).unwrap();
            r[1] == "\"Charlie\""
        })
        .map(|i| result.get_row(i).unwrap());

    assert!(charlie_row.is_some());
    let cr = charlie_row.unwrap();
    assert_eq!(cr[1], "\"Charlie\"");
    assert_eq!(cr[3], "null");
}

#[test]
fn test_join_column_collision_renames() {
    let left = make_int_df(vec![("id", vec![1i32]), ("name", vec![10])]);
    let right = make_int_df(vec![("id", vec![1i32]), ("name", vec![20])]);
    let result = left.join_inner(&right, "id", "id").unwrap();
    let cols = result.get_column_names();
    assert!(cols.contains(&"name"));
    assert!(cols.contains(&"name_right"));
}

#[test]
fn test_join_error_key_not_found() {
    let df = load_employees();
    let err = df.join_inner(&df, "missing", "employee_id");
    assert!(err.is_err());
    assert!(matches!(err, Err(CrossbowError::ColumnNotFound(_))));
}
