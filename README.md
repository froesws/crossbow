# Crossbow — Rust DataFrame Library

A lightweight, Arrow-backed DataFrame library for Rust. Built on Apache Arrow for zero-copy interoperability.

```toml
[dependencies]
crossbow = { path = "." }
```

## Quick Start

```rust
use crossbow::{DataFrame, Series};

// Create columns
let name   = Series::from("name",    vec!["Alice", "Bob", "Carol"]);
let age    = Series::from("age",     vec![25i32, 30, 28]);
let salary = Series::from("salary",  vec![75000i32, 82000, 91000]);

// Build a DataFrame
let df = DataFrame::new(vec![name, age, salary]).unwrap();
println!("{}", df);
// ┌─------──---──----───┐
// │  name   │ age │ salary │
// ├─------──---──----───┤
// │ "Alice" │ 25  │ 75000  │
// │ "Bob"   │ 30  │ 82000  │
// │ "Carol" │ 28  │ 91000  │
// └─------──---──----───┘
```

## Series — Named, Typed Columns

A `Series` wraps an Apache Arrow array, providing named columns of any supported type.

### Creating a Series

```rust
use crossbow::Series;

// From Rust vectors
let ages     = Series::from("age",     vec![18i32, 25, 42]);
let names    = Series::from("name",    vec!["Ana", "Bob"]);
let salaries = Series::from("salary",  vec![50000.0f64, 72000.0, 91000.0]);
let flags    = Series::from("active",  vec![true, false, true]);

// With nullable values
let maybe_ages = Series::from("age", vec![Some(18i32), None, Some(42)]);

// From an existing Arrow array
let array = arrow::array::Int32Array::from(vec![1, 2, 3]);
let series = Series::new("nums", std::sync::Arc::new(array));
```

### Inspecting a Series

```rust
let s = Series::from("vals", vec![10i32, 20, 30]);
assert_eq!(s.name(), "vals");
assert_eq!(s.len(), 3);
assert!(s.is_numeric());
assert!(!s.is_empty());
assert_eq!(s.dtype(), &arrow::datatypes::DataType::Int32);
```

### Accessing Values

```rust
let s = Series::from("vals", vec![10, 20, 30]);
assert_eq!(s.value_at(1).unwrap(), "20");

// Slicing
let first_two = s.slice(0, 2).unwrap();
assert_eq!(first_two.len(), 2);
```

### Arithmetic Operations

All element-wise. Nulls propagate — `null + x = null`.

```rust
let a = Series::from("a", vec![10i32, 20, 30]);
let b = Series::from("b", vec![1i32, 2, 3]);

let sum = a.add(&b).unwrap();       // [11, 22, 33]
let diff = a.subtract(&b).unwrap(); // [9, 18, 27]
let prod = a.multiply(&b).unwrap(); // [10, 40, 90]
let quot = a.divide(&b).unwrap();   // [10, 10, 10]
let rem  = a.modulo(&b).unwrap();   // [0, 0, 0]
```

### Comparisons

Returns a boolean `Series` usable as a filter mask.

```rust
use crossbow::{Numeric, Series};

let ages = Series::from("age", vec![18i32, 25, 42, 65]);
let mask = ages.greater_than(Numeric::int32(40)).unwrap();
// mask = [false, false, true, true]
```

### Null Handling

```rust
let s = Series::from("vals", vec![Some(10i32), None, Some(30)]);

let null_map = s.is_null().unwrap();    // [false, true, false]
let not_null = s.is_not_null().unwrap(); // [true, false, true]

let cleaned = s.drop_null().unwrap();  // [10, 30]
assert_eq!(cleaned.len(), 2);
```

### Aggregation

All aggregation ignores null values. `mean()`, `std()`, and `var()` return
`f64::NAN` for empty series or when fewer than 2 non-null values exist.

```rust
let s = Series::from("vals", vec![10i32, 20, 30, 40]);

s.sum().unwrap();                               // 100.0
s.mean().unwrap();                              // 25.0
s.min().unwrap();                               // Some(10.0)
s.max().unwrap();                               // Some(40.0)
s.count();                                      // 4
s.count_non_null().unwrap();                    // 4
s.std().unwrap();                               // sample std dev (NaN if < 2 values)
s.var().unwrap();                               // sample variance (NaN if < 2 values)
```

## DataFrame — Tabular Data

A `DataFrame` is a collection of `Series` (columns) of equal length with unique names.

### Creating a DataFrame

```rust
use crossbow::{DataFrame, Series};

let name   = Series::from("name",   vec!["Alice", "Bob"]);
let age    = Series::from("age",    vec![25i32, 30]);
let df = DataFrame::new(vec![name, age]).unwrap();

assert_eq!(df.shape(), (2, 2));
assert_eq!(df.get_column_names(), vec!["name", "age"]);
```

### Column Access

```rust
// Get a column by name
let age_col = df.select("age").unwrap();
assert_eq!(age_col.name(), "age");

// Iterate columns
for col in df.columns() {
    println!("{}: {:?}", col.name(), col.dtype());
}
```

### Row Access

```rust
let row = df.get_row(1).unwrap(); // ["\"Bob\"", "30"]
let rows = df.get_rows(&[0, 1]).unwrap();
```

### Column Management

```rust
// Add
let bonus = Series::from("bonus", vec![5000i32, 6000]);
let extended = df.add_column(bonus).unwrap();

// Rename
let renamed = df.rename_column("age", "years").unwrap();

// Remove
let slim = extended.remove_column("bonus").unwrap();
```

### Filtering

```rust
use crossbow::Numeric;

let ages = df.select("age").unwrap();
let mask = ages.greater_than(Numeric::int32(20)).unwrap();
let adults = df.filter_by_mask(&mask).unwrap();
// Only rows where age > 20
```

### Sorting

```rust
let by_age_asc  = df.sort_by_asc("age").unwrap();   // youngest first
let by_age_desc = df.sort_by_desc("age").unwrap();  // oldest first
let by_name     = df.sort_by_asc("name").unwrap();  // alphabetical
```

### Dates and Timestamps

```rust
// Date32 — days since Unix epoch
let dates = Series::from_date32("created", vec![19701, 19702, 19703]);
// Default: YYYY-MM-DD
assert_eq!(dates.value_at(0).unwrap(), "2023-12-10");

// Custom format
let br_dates = dates.with_date_format("DD/MM/YYYY");
assert_eq!(br_dates.value_at(0).unwrap(), "10/12/2023");

// Date64 / Timestamp — milliseconds since epoch, includes time
let stamps = Series::from_timestamp_ms("ts", vec![1702211696000i64]);
// Default: YYYY-MM-DD HH:mm:ss
assert!(stamps.value_at(0).unwrap().contains("2023-12-10 12:34:56"));

// Works with all core operations: filter, sort, shuffle, join, drop_null, fill_null
let df = DataFrame::new(vec![dates]).unwrap();
let sorted = df.sort_by("created", true).unwrap();   // chronological order

// Parse date strings from CSV into Date32
let strings = Series::from("str_dates", vec!["2023-12-10", "2024-01-15"]);
let parsed = strings.try_into_date32("YYYY-MM-DD").unwrap();
assert!(parsed.is_date());
```

### Shuffling

```rust
let df = read_csv("data.csv").unwrap();
let mixed = df.shuffle().unwrap();  // rows randomly reordered, values stay together
```

### GroupBy and Aggregation

```rust
let grouped = df.group_by("position").unwrap();

let counts  = grouped.count().unwrap();            // count per group
let sum     = grouped.sum("yield").unwrap();       // sum of yield per group
let avg     = grouped.mean("salary").unwrap();     // mean salary per group
let hi      = grouped.max("salary").unwrap();      // max salary per group
let lo      = grouped.min("salary").unwrap();      // min salary per group
```

### Joins

Three join types supported. The right-hand join key column is excluded from the result. If a right column name collides with a left column, `_right` is appended.

```rust
let employees  = read_csv("employees.csv").unwrap();
let departments = read_csv("departments.csv").unwrap();

// Inner join — only rows with matching keys on both sides
let inner = employees.join_inner(&departments, "dept_id", "dept_id").unwrap();

// Left join — all left rows, nulls for unmatched right
let left = employees.join_left(&departments, "dept_id", "dept_id").unwrap();

// Outer join — all rows from both sides, nulls where no match
let outer = employees.join_outer(&departments, "dept_id", "dept_id").unwrap();
```

| Join type | Left rows | Right rows | Unmatched |
|---|---|---|---|
| `join_inner` | matched only | matched only | dropped |
| `join_left` | all | matched | right cols null |
| `join_outer` | all | all | null-filled |

## I/O — Reading and Writing Files

### CSV

```rust
use crossbow::{read_csv, write_csv};

// Read with automatic type inference
let df = read_csv("employees.csv").unwrap();

// Write with headers
write_csv(&filtered, "output.csv").unwrap();
```

### Parquet

```rust
use crossbow::{read_parquet, write_parquet};

let df = read_parquet("data.parquet").unwrap();
write_parquet(&df, "copy.parquet").unwrap();
```

## Complete Example

```rust
use crossbow::{read_csv, write_csv, Numeric, Series};

fn main() {
    let df = read_csv("employees.csv").unwrap();

    // Filter: age >= 50
    let ages = df.select("age").unwrap();
    let mask = ages.greater_than(Numeric::int32(49)).unwrap();
    let seniors = df.filter_by_mask(&mask).unwrap();

    // Sort by salary descending
    let by_salary = seniors.sort_by_desc("salary").unwrap();

    // Group by position
    let grouped = df.group_by("position").unwrap();
    let avg_salary = grouped.mean("salary").unwrap();

    write_csv(&by_salary, "seniors_by_salary.csv").unwrap();
    write_csv(&avg_salary, "avg_salary_by_position.csv").unwrap();
}
```

## Error Handling

All fallible operations return `Result<_, CrossbowError>`:

```rust
match df.select("missing_column") {
    Ok(col) => println!("Found: {}", col.name()),
    Err(CrossbowError::ColumnNotFound(name)) => eprintln!("Missing: {name}"),
    Err(e) => eprintln!("Error: {e}"),
}
```

## Supported Types

| Arrow Type | Rust Type | Filter | Sort | Shuffle | Arithmetic | Aggregation |
|---|---|---|---|---|---|---|---|
| `Int32` | `i32` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Int64` | `i64` | ✓ | ✓ | ✓ | — | ✓ |
| `Float32` | `f32` | ✓ | ✓ | ✓ | — | ✓ |
| `Float64` | `f64` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Utf8` | `&str` / `String` | ✓ | ✓ | ✓ | — | — |
| `Boolean` | `bool` | ✓ | — | ✓ | — | — |
| `Date32` | `i32` days | ✓ | ✓ | ✓ | — | ✓ |
| `Date64` | `i64` ms | ✓ | ✓ | ✓ | — | ✓ |
| `Timestamp (ms)` | `i64` ms | ✓ | ✓ | ✓ | — | ✓ |

## License

MIT — see `src/lib.rs` header.
