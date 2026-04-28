//! Shared utility macros for type-dispatch and array column building.
//!
//! These macros eliminate repeated per-type match arms across filter, sort,
//! join, and null-handling code. They generate the correct Arrow builder,
//! downcast the source array, and fill values from a slice of indices.

/// Build an Arrow array from a source array and a slice of indices.
///
/// Each `usize` index selects one row from the source array. The macro
/// handles downcasting, builder allocation, null propagation, and finishing.
/// Utf8/StringBuilder uses `capacity * 16` as data capacity.
///
/// # Examples
///
/// ```
/// use crossbow::build_column;
/// use crossbow::Series;
/// use std::sync::Arc;
/// use arrow::array::Array;
///
/// let source = Series::from("src", vec![1i32, 2, 3, 4, 5]);
/// let indices = vec![0usize, 2, 4];
/// let n = indices.len();
/// let built = build_column!(
///     source.data().as_ref(),
///     &indices,
///     arrow::array::Int32Builder,
///     arrow::array::Int32Array,
///     n
/// );
/// ```
#[macro_export]
macro_rules! build_column {
    ($arr:expr, $indices:expr, $builder_ty:ty, $arr_downcast:ty, $cap:expr) => {{
        type B = $builder_ty;
        let a = $arr.as_any().downcast_ref::<$arr_downcast>().unwrap();
        let mut b = <B>::with_capacity($cap);
        for &idx in $indices {
            if a.is_valid(idx) {
                b.append_value(a.value(idx));
            } else {
                b.append_null();
            }
        }
        Arc::new(b.finish()) as Arc<dyn Array>
    }};
}

/// Build an Arrow array from optional indices (Some = row, None = null).
///
/// # Examples
///
/// ```
/// use crossbow::build_column_opt;
/// use crossbow::Series;
/// use std::sync::Arc;
/// use arrow::array::Array;
///
/// let source = Series::from("src", vec![10i32, 20, 30]);
/// let indices = vec![Some(0usize), None, Some(2usize)];
/// let n = indices.len();
/// let built = build_column_opt!(
///     source.data().as_ref(),
///     &indices,
///     arrow::array::Int32Builder,
///     arrow::array::Int32Array,
///     n
/// );
/// ```
#[macro_export]
macro_rules! build_column_opt {
    ($arr:expr, $indices:expr, $builder_ty:ty, $arr_downcast:ty, $cap:expr) => {{
        type B = $builder_ty;
        let a = $arr.as_any().downcast_ref::<$arr_downcast>().unwrap();
        let mut b = <B>::with_capacity($cap);
        for &oi in $indices {
            match oi {
                Some(idx) => {
                    if a.is_valid(idx) { b.append_value(a.value(idx)); }
                    else { b.append_null(); }
                }
                None => b.append_null(),
            }
        }
        Arc::new(b.finish()) as Arc<dyn Array>
    }};
}

/// Build a String/Utf8 array from indices (2-arg capacity for String).
///
/// # Examples
///
/// ```
/// use crossbow::build_string_column;
/// use crossbow::Series;
/// use std::sync::Arc;
/// use arrow::array::Array;
///
/// let source = Series::from("src", vec!["a", "b", "c", "d"]);
/// let indices = vec![1usize, 3];
/// let n = indices.len();
/// let built = build_string_column!(
///     source.data().as_ref(),
///     &indices,
///     n
/// );
/// ```
#[macro_export]
macro_rules! build_string_column {
    ($arr:expr, $indices:expr, $cap:expr) => {{
        let a = $arr.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
        let mut b = arrow::array::StringBuilder::with_capacity($cap, $cap * 16);
        for &idx in $indices {
            if a.is_valid(idx) {
                b.append_value(a.value(idx));
            } else {
                b.append_null();
            }
        }
        Arc::new(b.finish()) as Arc<dyn Array>
    }};
}

/// Build a String/Utf8 array from optional indices.
///
/// # Examples
///
/// ```
/// use crossbow::build_string_column_opt;
/// use crossbow::Series;
/// use std::sync::Arc;
/// use arrow::array::Array;
///
/// let source = Series::from("src", vec!["x", "y", "z"]);
/// let indices = vec![Some(0usize), None, Some(2usize)];
/// let n = indices.len();
/// let built = build_string_column_opt!(
///     source.data().as_ref(),
///     &indices,
///     n
/// );
/// ```
#[macro_export]
macro_rules! build_string_column_opt {
    ($arr:expr, $indices:expr, $cap:expr) => {{
        let a = $arr.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
        let mut b = arrow::array::StringBuilder::with_capacity($cap, $cap * 16);
        for &oi in $indices {
            match oi {
                Some(idx) => {
                    if a.is_valid(idx) { b.append_value(a.value(idx)); }
                    else { b.append_null(); }
                }
                None => b.append_null(),
            }
        }
        Arc::new(b.finish()) as Arc<dyn Array>
    }};
}

/// Error helper: produces a consistent `TypeMismatch` with the expected array name.
///
/// # Examples
///
/// ```
/// use crossbow::expected_array;
///
/// let err = expected_array("Int32Array");
/// assert!(format!("{}", err).contains("Type mismatch"));
/// ```
pub fn expected_array(expected_type_name: &str) -> crate::error::CrossbowError {
    crate::error::CrossbowError::TypeMismatch(
        format!("Expected {}", expected_type_name)
    )
}
